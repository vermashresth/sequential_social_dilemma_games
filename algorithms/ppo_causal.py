import copy
import sys

import numpy as np
import scipy

# TODO(@evinitsky) put this in alphabetical order

from ray.rllib.agents.ppo.ppo_policy import PPOLoss, BEHAVIOUR_LOGITS, \
    KLCoeffMixin, setup_config, clip_gradients, \
    kl_and_loss_stats, ValueNetworkMixin, vf_preds_and_logits_fetches, postprocess_ppo_gae
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, choose_policy_optimizer, \
    validate_config, update_kl, warn_about_bad_reward_scales
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule, ACTION_LOGP
from ray.rllib.utils import try_import_tf
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

from algorithms.common_funcs import setup_moa_loss, causal_fetches, setup_causal_mixins, get_causal_mixins, \
    causal_postprocess_trajectory, CAUSAL_CONFIG

tf = try_import_tf()

POLICY_SCOPE = "func"

CAUSAL_CONFIG.update(DEFAULT_CONFIG)


def loss_with_moa(policy, model, dist_class, train_batch):
    # you need to override this bit to pull out the right bits from train_batch
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    if policy.model.causal:
        moa_loss = setup_moa_loss(logits, model, policy, train_batch)
        policy.moa_loss = moa_loss.total_loss

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(
            train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOLoss(
        policy.action_space,
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[BEHAVIOUR_LOGITS],
        train_batch[ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"],
        model_config=policy.config["model"])
    if policy.model.causal:
        policy.loss_obj.loss += moa_loss.total_loss
    return policy.loss_obj.loss


def extra_fetches(policy):
    """Adds value function, logits, moa predictions of counterfactual actions to experience train_batches."""
    ppo_fetches = vf_preds_and_logits_fetches(policy)
    if policy.model.causal:
        ppo_fetches.update(causal_fetches(policy))
    return ppo_fetches


def extra_stats(policy, train_batch):
    base_stats = kl_and_loss_stats(policy, train_batch)
    if policy.model.causal:
        base_stats["total_influence"] = train_batch["total_influence"]
        base_stats['reward_without_influence'] = train_batch['reward_without_influence']
        base_stats['moa_loss'] = policy.moa_loss / policy.moa_weight
    return base_stats


def postprocess_ppo_causal(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""
    if policy.model.causal:
        sample_batch = causal_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, sample_batch)
    return batch


def build_model(policy, obs_space, action_space, config):
    _, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        logit_dim,
        config["model"],
        name=POLICY_SCOPE,
        framework="tf")

    return policy.model


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    if policy.model.causal:
        setup_causal_mixins(policy, obs_space, action_space, config)


CausalMOA_PPOPolicy = build_tf_policy(
    name="CausalTFPolicy",
    get_default_config=lambda: CAUSAL_CONFIG,
    loss_fn=loss_with_moa,
    make_model=build_model,
    stats_fn=extra_stats,
    extra_action_fetches_fn=extra_fetches,
    postprocess_fn=postprocess_ppo_causal,
    gradients_fn=clip_gradients,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ] + get_causal_mixins())

CausalPPOMOATrainer = build_trainer(
    name="CausalMOAPPO",
    default_policy=CausalMOA_PPOPolicy,
    make_policy_optimizer=choose_policy_optimizer,
    default_config=CAUSAL_CONFIG,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=warn_about_bad_reward_scales)
