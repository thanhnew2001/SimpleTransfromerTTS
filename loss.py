from hyperparams import hp
import torch

class TTSLoss(torch.nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py"""
    def __init__(self):
        super(TTSLoss, self).__init__()
        
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(
        self, 
        mel_postnet_out, 
        mel_out, 
        gate_out, 
        mel_target, 
        gate_target
      ):      
        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_postnet_out, mel_target)

        gate_loss = self.bce_loss(gate_out, gate_target) * hp.r_gate

        return mel_loss + gate_loss

if __name__ == "__main__":
  loss = TTSLoss()
  print(loss)  
