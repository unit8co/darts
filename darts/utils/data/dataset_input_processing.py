from abc import ABCMeta

from darts.utils.data.encoders import (
    CyclicPastEncoder,
    CyclicFutureEncoder,
    PositionalPastEncoder,
    PositionalFutureEncoder
)


class InputProcessor(ABCMeta):
    """this meta class should be applied to the outermost dataset load class"""
    def __call__(cls, *args, **kwargs):
        cls.process_input(*args, **kwargs)
        return super(InputProcessor, cls).__call__(*args, **kwargs)

    def process_input(cls, *args, **kwargs):
        add_cyclic_encoder = 'month'
        add_positional_encoder = None

        kwargs['add_cyclic_encoder'] = 'month'
        kwargs['add_positional_encoder'] = False
        # dummy version of what it should be

        add_encoders = list()
        add_encoders.append(kwargs.get('add_cyclic_encoder', None))
        # add_encoders.append(kwargs.get('add_cyclic_encoder', None))
        add_encoders.append(kwargs.get('add_positional_encoder', None))

        if not any(add_encoders):
            return args, kwargs
        all_encoders = [
            CyclicFutureEncoder,
            # CyclicFutureEncoder,
            PositionalFutureEncoder,
        ]
        icl, ocl = kwargs['input_chunk_length'], kwargs['output_chunk_length']
        encoders = [encoder(icl, ocl, add_enc) for add_enc, encoder in zip(add_encoders, all_encoders) if add_enc]

        for i in range(len(kwargs['future_covariates'])):
            print(kwargs['future_covariates'][i].n_components)
            for encoder in encoders:
                kwargs['future_covariates'][i] = encoder.encode_train(
                    idx=0,
                    target=kwargs['target_series'][i],
                    covariate=kwargs['future_covariates'][i]
                )
            print(kwargs['future_covariates'][i].n_components)
        return args, kwargs