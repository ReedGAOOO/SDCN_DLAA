def gat(gw,
        feature,
        hidden_size,
        activation,
        name,
        dist_feat=None,
        num_heads=4,
        feat_drop=0.2,
        attn_drop=0.2,
        is_test=False):
    """Implementation of graph attention networks (GAT)
    Adapted from https://github.com/PaddlePaddle/PGL/blob/main/pgl/layers/conv.py.
    """

    def send_attention(src_feat, dst_feat, edge_feat):
        output = src_feat["left_a"] + dst_feat["right_a"]
        if 'dist_a' in edge_feat:
            output += edge_feat["dist_a"]
        output = L.leaky_relu(
            output, alpha=0.2)  # (num_edges, num_heads)
        return {"alpha": output, "h": src_feat["h"]}

    def reduce_attention(msg):
        alpha = msg["alpha"]  # lod-tensor (batch_size, seq_len, num_heads)
        h = msg["h"]
        alpha = paddle_helper.sequence_softmax(alpha)
        old_h = h
        h = L.reshape(h, [-1, num_heads, hidden_size])
        alpha = L.reshape(alpha, [-1, num_heads, 1])
        if attn_drop > 1e-15:
            alpha = L.dropout(
                alpha,
                dropout_prob=attn_drop,
                is_test=is_test,
                dropout_implementation="upscale_in_train")
        h = h * alpha
        h = L.reshape(h, [-1, num_heads * hidden_size])
        h = L.lod_reset(h, old_h)
        return L.sequence_pool(h, "sum")

    if feat_drop > 1e-15:
        feature = L.dropout(
            feature,
            dropout_prob=feat_drop,
            is_test=is_test,
            dropout_implementation='upscale_in_train')
        if dist_feat:
            dist_feat = L.dropout(
                dist_feat,
                dropout_prob=feat_drop,
                is_test=is_test,
                dropout_implementation='upscale_in_train')

    ft = L.fc(feature,
              hidden_size * num_heads,
              bias_attr=False,
              param_attr=fluid.ParamAttr(name=name + '_weight'))
    left_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_l_A')
    right_a = L.create_parameter(
        shape=[num_heads, hidden_size],
        dtype='float32',
        name=name + '_gat_r_A')
    reshape_ft = L.reshape(ft, [-1, num_heads, hidden_size])
    left_a_value = L.reduce_sum(reshape_ft * left_a, -1)
    right_a_value = L.reduce_sum(reshape_ft * right_a, -1)
    efeat_list = [] # If dist_feat does not exist, efeat_list remains an empty list, which means that in this case the attention calculation does not include any edge feature influence.

    if dist_feat:
        fd = L.fc(dist_feat,
                  size=hidden_size * num_heads,
                  bias_attr=False,
                  param_attr=fluid.ParamAttr(name=name + '_fc_eW'))
        dist_a = L.create_parameter(
            shape=[num_heads, hidden_size],
            dtype='float32',
            name=name + '_gat_d_A')
        fd = L.reshape(fd, [-1, num_heads, hidden_size])
        dist_a_value = L.reduce_sum(fd * dist_a, -1)
        efeat_list = [('dist_a', dist_a_value)]

    msg = gw.send(
        send_attention,
        nfeat_list=[("h", ft), ("left_a", left_a_value),
                    ("right_a", right_a_value)], efeat_list=efeat_list)
    output = gw.recv(msg, reduce_attention)

    output = L.reshape(output, [-1, num_heads, hidden_size])
    output = L.reduce_mean(output, dim=1)
    num_heads = 1

    bias = L.create_parameter(
        shape=[hidden_size * num_heads],
        dtype='float32',
        is_bias=True,
        name=name + '_bias')
    bias.stop_gradient = True
    output = L.elementwise_add(output, bias, act=activation)
    return output