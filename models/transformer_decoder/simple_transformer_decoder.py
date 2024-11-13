import torch
import torch.nn as nn

import torch.nn.functional as F


# write a module. THe main model is Transformer decoder. The input to transformer decoder is queries.
# keys and values are coming from the image features of shape (B*NCAM, P, 192).
# we Need to add positional embedding (learnable) to the image features from different cams.
# the queries are initializzed with a learnable token. The output of the transformer decoder is of shape (B*NCAM, N, 256)
# the output is then passed through two MLPs to get mask embedding (192, numqueries) and class prediction (num_queries, 2)

class SimpleTransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_queries = args.num_queries

        H, W = args.resize_to
        P = H//16 * W//16
        self.cam_pos_embed = nn.Parameter(torch.randn(1, 6, 1, 192)*0.02)
        self.img_pos_embed = nn.Parameter(torch.randn(1, 1, P, 192)*0.02)

        self.queries = nn.Parameter(torch.randn(1, self.num_queries, 192)*0.02)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=192, nhead=8),
            num_layers=6
        )

        self.bev_latent_dim = args.bev_latent_dim
        self.bev_proj =  nn.Conv2d(
                                        self.bev_latent_dim, 
                                        192,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                    )
        # nn.Linear(self.bev_latent_dim, 192)

        self.mask_mlp = MLP(192, 192, hidden_dim=256, residual=False)
        self.class_mlp = MLP(192, 2, hidden_dim=256, residual=False)
        

    def forward(self, image_features, bev_features, mask=None):
        '''
            image_features: (B, NCAM, P, 192)
            bev_features: (B, C, H, W)
        '''

        B, NCAM, P, _ = image_features.shape

        # add cam positional embedding to image features
        cam_pos_embed = self.cam_pos_embed.expand(B, -1, P, -1)
        image_features = image_features + cam_pos_embed

        # add image positional embedding to image features
        img_pos_embed = self.img_pos_embed.expand(B, NCAM, -1, -1)
        image_features = image_features + img_pos_embed

        image_features = image_features.view(B, NCAM*P, 192)     # (B, NCAM*P, 192)
        image_features = image_features.permute(1, 0, 2)         # (NCAM*P, B, 192)

        queries = self.queries.expand(B, -1, -1)                      # (B, Q, 192)
        queries = queries.permute(1, 0, 2)                            # (Q, B, 192)

        # pass through transformer decoder
        output = self.transformer_decoder(queries, image_features)      # (Q, B, 192)
        output = output.permute(1, 0, 2)                                # (B, Q, 192)
        
        # pass through MLPs
        mask_embedding = self.mask_mlp(output)                    # (B, Q, 192)
        class_prediction = self.class_mlp(output)                 # (B, Q, 2)

        bev_features = bev_features     # (B, C, H, W)
        bev_features = self.bev_proj(bev_features)                # (B, 192, H, W)

        # mask embedding * bev features
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embedding, bev_features)

        output = {
            'pred_masks': outputs_mask,
            'pred_logits': class_prediction
        }
        return output
    



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out    
