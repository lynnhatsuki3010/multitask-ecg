import torch
from src.models.backbones import MultiBranchTransformerBackbone, ConductionLeadGroupEncoder
from src.models.ecg_multitask import ECGMultiTaskModel

x = torch.randn(4, 12, 5000)

# Test ConductionLeadGroupEncoder standalone
enc = ConductionLeadGroupEncoder(out_dim=128, dropout=0.1)
out = enc(x)
print("ConductionLeadGroupEncoder:", out.shape)  # expect (4, 128)

# Test heterogeneous backbone
bb = MultiBranchTransformerBackbone(
    d_model=256, num_shared_layers=2, num_expert_layers=2,
    arrhy_layers=2, arrhy_nhead=8, arrhy_ffn=512,
    mi_layers=2,    mi_nhead=8,    mi_ffn=512,
    cond_layers=2,  cond_nhead=8,  cond_ffn=512,
    cd_lead_out_dim=128,
)
res = bb(x)
print("arrhy_global    :", res["arrhy_global"].shape)
print("cond_global     :", res["cond_global"].shape)
print("cond_lead_feats :", res["cond_lead_feats"].shape)

# Test full ECGMultiTaskModel
model = ECGMultiTaskModel(
    backbone=bb,
    num_arrhythmia_labels=5,
    num_mi_labels=4,
    num_conduction_labels=4,
    hrv_enabled=False,
    num_hrv_targets=3,
    head_hidden_dim=128,
    mi_branch_dim=128,
    dropout=0.15,
)
outs = model(x)
print("arrhythmia output:", outs["arrhythmia"].shape)
print("mi output        :", outs["mi"].shape)
print("conduction output:", outs["conduction"].shape)
print("All OK!")
