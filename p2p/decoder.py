from torch import nn

class Decoder(nn.Module):

    def __init__(self, input_feature_sizes, feature_size):

        super(Decoder, self).__init__()

        self.decoder_levels = nn.ModuleList()
        for input_f_sz in input_feature_sizes:
            curr_decoder_level = nn.ModuleList(
                [nn.Conv2d(input_f_sz, feature_size, kernel_size=1, stride=1, padding=0),
                 nn.Upsample(scale_factor=2, mode='nearest'),
                 nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)]
            )
            self.decoder_levels.append(curr_decoder_level)


    def forward(self, inputs):

        assert len(inputs) == len(self.decoder_levels)

        out = []
        prev_level_upsampled = None
        for lvl, curr_decoder_level in enumerate(self.decoder_levels[::-1]):
            conv_1, upsample, conv_2 = curr_decoder_level
            x = conv_1(inputs[-(lvl + 1)])
            if prev_level_upsampled is None:
                x_upsampled = upsample(x)
            else:
                x_upsampled = upsample(x + prev_level_upsampled)
            prev_level_upsampled = x_upsampled
            out.append(conv_2(x))

        return out[::-1]
