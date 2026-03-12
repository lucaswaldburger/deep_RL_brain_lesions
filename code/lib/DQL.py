import logging
import os
import random
from collections import namedtuple

import numpy as np
import psutil
import tensorflow as tf
from PIL import Image

from config import DEFAULT_CONFIG
from lib.Agent import ObjLocaliser
from lib.DNN import VALID_ACTIONS, ModelParametersCopier, Estimator, StateProcessor, make_epsilon_greedy_policy
from lib.ReadData import extractData
from lib.session_utils import run_episode

logger = logging.getLogger(__name__)


def evaluate(tmp, state_processor, policy, sess, num_of_proposal=15):
    """Evaluate a network on a single image.

    Args:
        tmp: A tuple of (image_dict, target_dict).
        state_processor: A StateProcessor instance.
        policy: An epsilon-greedy policy function.
        sess: TensorFlow session.
        num_of_proposal: Number of episodes for evaluation.

    Returns:
        Mean precision for the input image.
    """
    img = tmp[0]
    target = tmp[1]
    succ = 0

    im2 = Image.frombytes("RGB", (img['image_width'], img['image_height']), img['image'])
    env = ObjLocaliser(np.array(im2), target)

    for _ in range(num_of_proposal):
        success, _ = run_episode(env, np.array(im2), state_processor, policy, sess)
        if success:
            succ += 1

    return float(succ) / num_of_proposal


def DQL(num_episodes, replay_memory_size, replay_memory_init_size,
        update_target_estimator_every, discount_factor, epsilon_start,
        epsilon_end, epsilon_decay_steps, category, model_name, type):
    """Build and train the deep Q-network.

    Args:
        num_episodes: Number of episodes per image.
        replay_memory_size: Max replay memory capacity.
        replay_memory_init_size: Initial replay memory population size.
        update_target_estimator_every: Steps between target network updates.
        discount_factor: Discount factor for future rewards.
        epsilon_start: Epsilon schedule start value.
        epsilon_end: Epsilon schedule end value.
        epsilon_decay_steps: Number of steps for epsilon decay.
        category: List of image categories for training.
        model_name: Name for saving the trained model.
        type: Tumor type ('HGG' or 'LGG').
    """
    cfg = DEFAULT_CONFIG

    tf.reset_default_graph()

    experiment_dir = os.path.abspath(
        os.path.join(cfg.experiments_dir, category[0] + '_' + type[0] + "_experiments", model_name),
    )

    global_step = tf.Variable(0, name='global_step', trainable=False)

    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")
    state_processor = StateProcessor()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        replay_memory = []
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)
        current_process = psutil.Process()

        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        report_path = os.path.join(experiment_dir, "report")
        best_model_dir = os.path.join(experiment_dir, "bestModel")
        best_model_path = os.path.join(best_model_dir, "model")

        for d in [checkpoint_dir, report_path, best_model_dir]:
            if not os.path.exists(d):
                os.makedirs(d)

        report_file = open(
            os.path.join(report_path, category[0] + '_' + type[0] + "_log.txt"), 'w',
        )

        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            logger.info("Loading model checkpoint %s...", latest_checkpoint)
            saver.restore(sess, latest_checkpoint)

        total_t = sess.run(tf.contrib.framework.get_global_step())
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        policy = make_epsilon_greedy_policy(q_estimator, len(VALID_ACTIONS))

        episode_counter = 0
        best_pre = 0
        eval_pre = []
        eval_set = []
        batch_size = cfg.batch_size
        done = False
        num_located = 0

        logger.info("Category: %s, Type: %s", category, type)

        for indx, tmp in enumerate(extractData(category, type, "train", batch_size)):
            img = tmp[0]
            target = tmp[1]

            if len(eval_set) < cfg.eval_set_size:
                logger.info("Populating evaluation set...")
                eval_set.append(tmp)
            else:
                if indx % cfg.eval_every_n_images == 0:
                    logger.info("Evaluation started...")
                    for tmp2 in eval_set:
                        eval_pre.append(evaluate(tmp2, state_processor, policy, sess))
                        if len(eval_pre) > cfg.eval_set_size - 1:
                            mean_pre = np.mean(eval_pre)
                            logger.info("Evaluation mean precision: %s", mean_pre)
                            report_file.write("Evaluation mean precision: {}\n".format(mean_pre))

                            episode_summary = tf.Summary()
                            episode_summary.value.add(simple_value=mean_pre, tag="episode/eval_acc")
                            q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                            q_estimator.summary_writer.flush()

                        if np.mean(eval_pre) > best_pre:
                            best_pre = np.mean(eval_pre)
                            logger.info("Best model changed with mean precision: %s", best_pre)
                            report_file.write("Best model changed with mean precision: {}\n".format(best_pre))
                            saver.save(tf.get_default_session(), best_model_path)
                        eval_pre = []

                im2 = Image.frombytes("RGB", (img['image_width'], img['image_height']), img['image'])
                env = ObjLocaliser(np.array(im2), target)
                logger.info("Image %d is being loaded: %s", indx, img['image_filename'])
                report_file.write("Image{} is being loaded: {}\n".format(indx, img['image_filename']))

                if len(replay_memory) < replay_memory_init_size:
                    logger.info("Populating replay memory...")

                    env.Reset(np.array(im2))
                    state = env.wrapping()
                    state = state_processor.process(sess, state)
                    state = np.stack([state] * 4, axis=2)

                    for _ in range(replay_memory_init_size):
                        action_probs, _ = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps - 1)])
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        reward = env.takingActions(VALID_ACTIONS[action])
                        next_state = env.wrapping()

                        done = (action == 10)

                        next_state = state_processor.process(sess, next_state)
                        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                        replay_memory.append(Transition(state, action, reward, next_state, done))
                        state = next_state

                        if done:
                            env.Reset(np.array(im2))
                            state = env.wrapping()
                            state = state_processor.process(sess, state)
                            state = np.stack([state] * 4, axis=2)
                        else:
                            state = next_state

                for i_episode in range(num_episodes):
                    saver.save(tf.get_default_session(), checkpoint_path)

                    env.Reset(np.array(im2))
                    state = env.wrapping()
                    state = state_processor.process(sess, state)
                    state = np.stack([state] * 4, axis=2)
                    loss = None
                    t = 0
                    action = 0
                    e = 0
                    r = 0

                    while (action != 10) and (t < cfg.max_actions_per_episode):
                        epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

                        if total_t % update_target_estimator_every == 0:
                            estimator_copy.make(sess)
                            logger.info("Copied model parameters to target network.")

                        action_probs, qs = policy(sess, state, epsilon)
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        reward = env.takingActions(VALID_ACTIONS[action])
                        next_state = env.wrapping()
                        done = (action == 10)

                        next_state = state_processor.process(sess, next_state)
                        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

                        if len(replay_memory) == replay_memory_size:
                            replay_memory.pop(0)

                        replay_memory.append(Transition(state, action, reward, next_state, done))

                        samples = random.sample(replay_memory, batch_size)
                        states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                        q_values_next = target_estimator.predict(sess, next_states_batch)
                        targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

                        states_batch = np.array(states_batch)
                        loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

                        msg = "Step {} ({}) @ Episode {}/{}, action {}, reward {}, loss: {}".format(
                            t, total_t, i_episode + 1, num_episodes, action, reward, loss,
                        )
                        logger.info(msg)
                        report_file.write(msg + "\n")

                        if reward == 3:
                            num_located += 1

                        state = next_state
                        t += 1
                        total_t += 1
                        e = e + loss
                        r = r + reward

                    episode_counter += 1

                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
                    episode_summary.value.add(simple_value=r, tag="episode/reward")
                    episode_summary.value.add(simple_value=t, tag="episode/length")
                    episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
                    episode_summary.value.add(simple_value=current_process.memory_percent(), tag="system/v_memory_usage_percent")
                    q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                    q_estimator.summary_writer.flush()

                    msg = "Episode Reward: {} Episode Length: {}".format(r, t)
                    logger.info(msg)
                    report_file.write(msg + "\n")

        report_file.close()
        logger.info("Number of correctly located objects: %d", num_located)
