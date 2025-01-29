from __init__ import *
import torch as t
from transformers import AutoModel, AutoTokenizer


class intervention_model(t.nn.Module):
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
    
        neuron_dim = (1, 768)
        self.l4_mask = t.nn.Parameter(
            t.zeros(neuron_dim, device=self.args.device), requires_grad=True
        )
        self.l4_mask = self.l4_mask.to(self.args.device)

    def forward(self, source_ids, base_ids, temperature): 
        l4_mask_sigmoid = t.sigmoid(self.l4_mask / temperature)
        with self.model.trace() as tracer:

            with tracer.invoke(source_ids) as runner:
                vector_source = self.model.transformer.h[
                    self.layer_intervened
                ].output[0][0]

            with tracer.invoke(base_ids) as runner_:
                intermediate_output = (
                    self.model.transformer.h[self.layer_intervened]
                    .output[0]
                    .clone()
                )
                intermediate_output = (1 - l4_mask_sigmoid) * intermediate_output[
                    :, self.intervened_token_idx, :
                ] + l4_mask_sigmoid * vector_source[:, self.intervened_token_idx, :]
                assert (
                    intermediate_output.squeeze(1).shape
                    == vector_source[:, self.intervened_token_idx, :].shape
                    == torch.Size([self.batch_size, 768])
                )
                self.model.transformer.h[self.layer_intervened].output[0][0][
                    :, self.intervened_token_idx, :
                ] = intermediate_output.squeeze(1)
                # self.model.transformer.h[self.layer_intervened].output[0][0][:,self.intervened_token_idx,:] = vector_source[:,self.intervened_token_idx,:]

                intervened_base_predicted = self.model.lm_head.output.argmax(
                    dim=-1
                ).save()
                intervened_base_output = self.model.lm_head.output.save()

        predicted_text = []
        for index in range(intervened_base_output.shape[0]):
            predicted_text.append(
                self.model.tokenizer.decode(
                    intervened_base_output[index].argmax(dim=-1)
                ).split()[-1]
            )
            
        return intervened_base_output, predicted_text


def parser_arguments():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="mps")
    
    args = parser.parse_args()
    
    return args

def main():
    args = parser_arguments()
    model, tokenizer = config(args).forward()
    model = model(model, tokenizer, args)


if __name__ == "__main__":
    main()