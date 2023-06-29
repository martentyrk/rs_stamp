import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time


class NN(object):
    """
    The memory network with context attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        if config != None:
            # the config.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            self.is_update_lr = config['update_lr']
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            if self.is_update_lr:
                self.decay_steps = config['decay_steps']
                self.decay_rate = config['decay_rate']
        # the optimize set.
        self.global_step = None  # the step counter.
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.

    def optimize_normal(self, loss, params):
        '''
        optimize
        loss: the loss.
        params: the params need to be optimized.
        '''
        # the optimize.
        self.global_step = tf.Variable(0, name='global_step')
        if self.is_update_lr:
            self.lr = self.update_lr()
        else:
            self.lr = tf.Variable(self.init_lr, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads_and_vars = self.optimizer.compute_gradients(loss, params)
        if self.max_grad_norm != None:
            clipped_grads_and_vars = [
                (tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) for gv in grads_and_vars
            ]
        else:
            clipped_grads_and_vars = grads_and_vars
        inc = self.global_step.assign_add(1)
        optimize = None
        with tf.control_dependencies([inc]):
            optimize = self.optimizer.apply_gradients(
                clipped_grads_and_vars)
        return optimize

    def update_lr(self):
        '''
        update the learning rate.
        '''
        lr = tf.train.exponential_decay(
            self.init_lr,
            self.global_step / 28,
            self.decay_steps,
            self.decay_rate,
            staircase=True
        )
        return lr

    def save_model(self, sess, config, saver=None, val=False):
        suf = time.strftime("%Y%m%d%H%M", time.localtime())[4:]

        # For validation results, we save the best model
        if val == True:
            path = f"{config['model_save_path']}{config['model']}-{config['dataset']}-{config['k_folds']}folds-atk{config['cut_off']}-best_model.ckpt"
        # When testing, save the model with current timestamp
        else:
            path = f"{config['model_save_path']}{config['model']}-{config['dataset']}-{config['k_folds']}folds-atk{config['cut_off']}-{suf}.ckpt"
        if saver is not None:
            saver.save(sess, path)
        config['saved_model'] = path
        return config
        

