def wrap_attacker_trainer(base_trainer, config):
    '''Wrap the trainer for attack client.
    Args:
        base_trainer (core.trainers.GeneralTorchTrainer): the trainer that
        will be wrapped;
        config (federatedscope.core.configs.config.CN): the configure;

    :returns:
        The wrapped trainer; Type: core.trainers.GeneralTorchTrainer

    '''
    if config.attack.attack_method.lower() == 'gan_attack':
        from federatedscope.attack.trainer import wrap_GANTrainer
        return wrap_GANTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'gradascent':
        from federatedscope.attack.trainer import wrap_GradientAscentTrainer
        return wrap_GradientAscentTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'backdoor':
        from federatedscope.attack.trainer import wrap_backdoorTrainer
        return wrap_backdoorTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'gaussian_noise':
        from federatedscope.attack.trainer import wrap_GaussianAttackTrainer
        return wrap_GaussianAttackTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'sr_targeted_random_sasrec':
        ## Random SASRec Attack
        from federatedscope.attack.trainer import wrap_SrTargetedRandomAttackSasrecTrainer
        return wrap_SrTargetedRandomAttackSasrecTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'sr_targeted_segment_sasrec':
        ## Segment SASRec Attack
        from federatedscope.attack.trainer import wrap_SrTargetedSegmentAttackSasrecTrainer
        return wrap_SrTargetedSegmentAttackSasrecTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'sr_targeted_labelflip_sasrec':
        ## Label Flipping Attack
        from federatedscope.attack.trainer import wrap_SrTargetedLabelFlipAttackSasrecTrainer
        return wrap_SrTargetedLabelFlipAttackSasrecTrainer(base_trainer)
    #elif config.attack.attack_method.lower() == 'sr_targeted_coordinated_sasrec':
    #    ## coordinated attack using embeddings
    #    from federatedscope.attack.trainer import wrap_TestAttackTrainer
    #    return wrap_TestAttackTrainer(base_trainer)
    elif config.attack.attack_method.lower() == 'sr_targeted_smart_random_sasrec':
        ## Random Smart Label SASRec Attack
        from federatedscope.attack.trainer import wrap_SrTargetedSmartRandomAttackSasrecTrainer
        return wrap_SrTargetedSmartRandomAttackSasrecTrainer(base_trainer)
    
    else:
        raise ValueError('Trainer {} is not provided'.format(
            config.attack.attack_method))
