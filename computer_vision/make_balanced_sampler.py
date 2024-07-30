def make_balanced_sampler(y):
    classes , counts = y.unique(return_counts = True )
    weights = 1.0 / counts.float()
    sample_weights = weights[y.squeeze().long()]
    #build sampler with compute weights 
    generator  = torch.Generator()
    sampler = WeightedRandomSampler(
        weights = sample_weights , 
        num_samples= len(sample_weights), 
        generator= generator , 
        replacement= True
    )
    return sampler
