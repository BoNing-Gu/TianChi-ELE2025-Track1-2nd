import torch

from FireRedASR.models.module.conformer_encoder import ConformerEncoder
from FireRedASR.models.module.transformer_decoder import TransformerDecoder


class FireRedAsrAed(torch.nn.Module):
    @classmethod
    def from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super().__init__()
        self.sos_id = args.sos_id
        self.eos_id = args.eos_id

        self.encoder = ConformerEncoder(
            args.idim, args.n_layers_enc, args.n_head, args.d_model,
            args.residual_dropout, args.dropout_rate,
            args.kernel_size, args.pe_maxlen)

        self.decoder = TransformerDecoder(
            args.sos_id, args.eos_id, args.pad_id, args.odim,  # odim=7832
            args.n_layers_dec, args.n_head, args.d_model,
            args.residual_dropout, args.pe_maxlen)

    def transcribe(self, padded_input, input_lengths,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0):
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        nbest_hyps = self.decoder.batch_beam_search(
            enc_outputs, enc_mask,
            beam_size, nbest, decode_max_len,
            softmax_smoothing, length_penalty, eos_penalty)
        return nbest_hyps
    
    def forward(self, padded_input, input_lengths, padded_target, target_lengths):
        """
        Args:
            padded_input: (batch_size, max_input_len, feat_dim)
            input_lengths: (batch_size)
            padded_target: (batch_size, max_target_len)
            target_lengths: (batch_size)
        Returns:
            logits: (batch_size, max_target_len-1, vocab_size)
            encoder_outputs: (batch_size, max_input_len, d_model)
        """
        # Encoder forward
        enc_outputs, _, enc_mask = self.encoder(padded_input, input_lengths)
        
        # Decoder forward (teacher forcing during training)
        # # Remove the last token from target for decoder input
        # decoder_input = padded_target[:, :-1]
        # Remove the first token (sos) from target for loss calculation
        decoder_output = self.decoder(padded_target, enc_outputs, enc_mask)
        
        return decoder_output, enc_outputs
