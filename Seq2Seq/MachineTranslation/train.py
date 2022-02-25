import torch
from torch.utils.data import DataLoader

from Seq2Seq.MachineTranslation.Decoder import Decoder
from Seq2Seq.MachineTranslation.Encoder import Encoder
from Seq2Seq.MachineTranslation.preprocess import BOS, PAD, read_data
from Seq2Seq.MachineTranslation.translate import greedy_translate


def batch_loss(encoder, decoder, X, Y, loss, out_vocab):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    dec_state = decoder.begin_state(enc_state)  # 初始化decoder的隐藏状态
    dec_input = torch.Tensor([out_vocab.stoi[BOS]] * batch_size)  # dec_input形状为(batch_size)
    mask, num_not_pad_tokens = torch.ones(batch_size, ), 0  # mask的形状为(batch_size)
    loss_value = torch.Tensor([0.0])
    for y in Y.permute(1, 0):  # Y的形状是(batch_size, seq_len), y的形状则是(batch_size),对应一批某一时刻
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)  # 计算一个时间步
        loss_value += (mask * loss(dec_output, y.long())).sum()
        dec_input = y  # 强制教学,将这一步应该输出的真实标签作为下一步的输入
        num_not_pad_tokens += mask.sum().item()
        mask = mask * (y != out_vocab.stoi[PAD]).float()  # 将PAD作为输入计算的loss忽略
    return loss_value / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = torch.nn.CrossEntropyLoss(reduction='none')
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        loss_sum = 0.0
        for X, Y in data_iter:
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            batch_mean_loss = batch_loss(encoder, decoder, X, Y, loss, out_vocab)
            batch_mean_loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            loss_sum += batch_mean_loss.item()
        if (epoch + 1) % 10 == 0:
            print("epoch: %d, loss: %.3f" % (epoch + 1, loss_sum / len(data_iter)))


if __name__ == "__main__":
    embed_size = 64
    num_hidden = 64
    num_layers = 2
    attention_size = 10
    drop_prob = 0.5
    lr = 0.01
    batch_size = 2
    num_epochs = 50
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    encoder = Encoder(len(in_vocab), embed_size, num_hidden, num_layers, drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hidden, num_layers, attention_size, drop_prob)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs, out_vocab)

    input_seq = "ils regardent ."
    output_seq = greedy_translate(encoder, decoder, in_vocab, out_vocab, input_seq, max_seq_len)
    print(output_seq)
