from __future__ import absolute_import, division, print_function, unicode_literals

import typing
import json
import functools
import torch

class GlowDLRMOptions(typing.NamedTuple):
    dense_feature_size: int
    sparse_feature_size: int
    embedding_feature_sizes: typing.List[int]
    embedding_op: str
    bottom_mlp_arch: typing.List[int]
    top_mlp_arch: typing.List[int]
    activation_function: str
    interaction_op: str
    training_set_size: int
    indices_per_example: int
    mini_batch_size: int
    num_epochs: int
    learning_rate: float

class ParameterSet(typing.NamedTuple):
    emb: typing.List[typing.List[float]]
    bottom_weight: typing.List[typing.List[float]]
    bottom_bias: typing.List[typing.List[float]]
    top_weight: typing.List[typing.List[float]]
    top_bias: typing.List[typing.List[float]]

class MinibatchInput(typing.NamedTuple):
    dense: typing.List[float]
    offsets: typing.List[typing.List[int]]
    indices: typing.List[typing.List[int]]
    targets: typing.List[float]

class MinibatchOutput(typing.NamedTuple):
    output: typing.List[float]
    loss: float
    acc: float

class MinibatchData(typing.NamedTuple):
    input: MinibatchInput
    output: MinibatchOutput
    update: ParameterSet

class GlowReplayCreator(object):
    def __init__(self, args):
        embedding_feature_sizes = [int(e) for e in args.arch_embedding_size.split("-")]
        bottom_mlp_arch = [int(e) for e in args.arch_mlp_bot.split("-")]
        top_mlp_arch = [int(e) for e in args.arch_mlp_top.split("-")]

        self.opts = GlowDLRMOptions(
            dense_feature_size=bottom_mlp_arch[0],
            sparse_feature_size=args.arch_sparse_feature_size,
            embedding_feature_sizes=embedding_feature_sizes,
            embedding_op="sls",
            bottom_mlp_arch=bottom_mlp_arch[1:],
            top_mlp_arch=top_mlp_arch,
            activation_function=args.activation_function,
            interaction_op=args.arch_interaction_op,
            training_set_size=args.data_size,
            indices_per_example=args.num_indices_per_lookup,
            mini_batch_size=args.mini_batch_size,
            num_epochs=args.nepochs,
            learning_rate=args.learning_rate,
        )

        self.emb_init = []
        self.bot_weight_init = []
        self.bot_bias_init = []
        self.top_weight_init = []
        self.top_bias_init = []

        self.minibatch_data = []

        self.filename = args.save_glow_replay

    def pull(self, t):
        return t.detach().cpu().numpy()

    def flatten(self, t, transpose=False):
        if transpose:
            t = t.transpose()

        return t.reshape((t.size)).tolist()

    def pull_and_flatten(self, t, transpose=False):
        pull_t = self.pull(t)
        flatten_t = self.flatten(pull_t, transpose)
        return flatten_t

    def get_emb_weights(self, emb_l):
        result = []
        for emb, size in zip(emb_l, self.opts.embedding_feature_sizes):
            for param in emb.parameters():
                proc_param = self.pull_and_flatten(param)
                assert(size * self.opts.sparse_feature_size == len(proc_param))
                result.append(proc_param)

        return result

    def get_mlp_weights_and_biases(self, mlp):
        weights = []
        biases = []

        for layer in mlp:
            for param in layer.parameters():
                proc_param = self.pull_and_flatten(param, transpose=True)

                if len(param.shape) == 2:
                    weights.append(proc_param)
                elif len(param.shape) == 1:
                    biases.append(proc_param)
                else:
                    assert("Unrecognized parameter")

        return weights, biases

    def add_embedding_initializers(self, emb_l):
        self.emb_init = self.get_emb_weights(emb_l)

    def add_bottom_mlp_initializers(self, mlp):
        weight_init, bias_init = self.get_mlp_weights_and_biases(mlp)

        assert(len(weight_init) == len(bias_init))
        assert(len(weight_init) == len(self.opts.bottom_mlp_arch))


        bottom_mlp_arch_with_dense = [self.opts.dense_feature_size] + self.opts.bottom_mlp_arch

        for i in range(len(bottom_mlp_arch_with_dense)-1):
            n_i = bottom_mlp_arch_with_dense[i]
            n_o = bottom_mlp_arch_with_dense[i+1]

            assert(len(weight_init[i]) == (n_i * n_o))
            assert(len(bias_init[i]) == (1 * n_o))

        self.bot_weight_init = weight_init
        self.bot_bias_init = bias_init

    def add_top_mlp_initializers(self, mlp):
        weight_init, bias_init = self.get_mlp_weights_and_biases(mlp)

        num_fea = len(self.opts.embedding_feature_sizes) + 1
        m_den_out = self.opts.bottom_mlp_arch[-1]
        num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

        top_mlp_arch_with_int = [num_int] + self.opts.top_mlp_arch

        assert(len(weight_init) == len(bias_init))
        assert(len(weight_init) == len(top_mlp_arch_with_int)-1)

        for i in range(len(top_mlp_arch_with_int)-1):
            n_i = top_mlp_arch_with_int[i]
            n_o = top_mlp_arch_with_int[i+1]

            assert(len(weight_init[i]) == (n_i * n_o))
            assert(len(bias_init[i]) == (1 * n_o))

        self.top_weight_init = weight_init
        self.top_bias_init = bias_init

    def add_initial_state(self, dlrm):
        self.add_embedding_initializers(dlrm.emb_l)
        self.add_bottom_mlp_initializers(dlrm.bot_l)
        self.add_top_mlp_initializers(dlrm.top_l)

    def add_minibatch_data_and_model_state(
        self,
        dense,
        offsets,
        indices,
        targets,
        outputs,
        loss,
        acc,
        dlrm,
    ):
        offsets = [offsets[i] for i in range(offsets.shape[0])]

        proc_dense = self.pull_and_flatten(dense)

        proc_offsets = []
        proc_indices = []

        for (off, ind) in zip(offsets, indices):
            proc_offsets.append(self.pull_and_flatten(off))
            proc_indices.append(self.pull_and_flatten(ind))

        proc_outputs = self.flatten(outputs)
        proc_loss = self.flatten(loss)[0]
        proc_targets = self.flatten(targets)

        mb_input = MinibatchInput(
            dense=proc_dense,
            offsets=proc_offsets,
            indices=proc_indices,
            targets=proc_targets,
        )

        mb_output = MinibatchOutput(
            output=proc_outputs,
            loss=proc_loss,
            acc=acc,
        )

        bw, bb = self.get_mlp_weights_and_biases(dlrm.bot_l)
        tw, tb = self.get_mlp_weights_and_biases(dlrm.top_l)

        mb_update = ParameterSet(
            emb=self.get_emb_weights(dlrm.emb_l),
            bottom_weight=bw,
            bottom_bias=bb,
            top_weight=tw,
            top_bias=tb,
        )

        self.minibatch_data.append(
            MinibatchData(
                input=mb_input,
                output=mb_output,
                update=mb_update,
            )
        )

    def save(self):
        save_dict = {}
        save_dict["conf"] = self.opts._asdict()
        init = ParameterSet(
            emb=self.emb_init,
            bottom_weight=self.bot_weight_init,
            bottom_bias=self.bot_bias_init,
            top_weight=self.top_weight_init,
            top_bias=self.top_bias_init,
        )

        save_dict["init"] = init._asdict()
        save_dict["data"] = [{"input": n.input._asdict(), "output": n.output._asdict(), "update": n.update._asdict()} for n in self.minibatch_data]

        with open(self.filename, "w") as f:
            json.dump(save_dict, f)

    def print_opts(self):
        print(json.dumps(self.opts._asdict()))
