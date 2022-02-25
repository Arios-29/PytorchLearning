import torch

from Seq2Seq.MachineTranslation.preprocess import EOS, PAD, BOS


def greedy_translate(encoder, decoder, in_vocab, out_vocab, input_seq, max_seq_len):
    in_tokens = input_seq.split(" ")
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.Tensor([[in_vocab.stoi[word] for word in in_tokens]])  # (1, max_seq_len)
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.Tensor([out_vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == EOS:
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens

