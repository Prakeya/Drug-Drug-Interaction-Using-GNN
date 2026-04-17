import re
import traceback

try:
    with open("scripts/train_model.py", "r") as f:
        content = f.read()

    # 1. Add focal loss class
    focal_loss_class = """
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def collate_fn"""
    content = content.replace("def collate_fn", focal_loss_class)

    # 2. Replace CrossEntropy args processing
    old_criterion = "print(f\"QT loss boost factor: {args.qt_loss_boost:.2f}\")\\n\\t\\tcriterion = None"
    new_criterion = "print(f\"Using FocalLoss with gamma=2.0 and Class Weights.\")\\n\\t\\tcriterion = FocalLoss(weight=class_weights, gamma=2.0)"
    content = content.replace(old_criterion, new_criterion)

    # 3. Replace the loss calculation loop
    old_loss_calc = """			if task_type == "multiclass":
				y_long = y.long()
				per_sample_loss = F.cross_entropy(
					logits,
					y_long,
					weight=class_weights,
					reduction="none",
				)
				sample_boost = torch.ones_like(per_sample_loss)
				sample_boost = torch.where(
					y_long == qt_idx,
					torch.full_like(sample_boost, float(args.qt_loss_boost)),
					sample_boost,
				)
				loss = (per_sample_loss * sample_boost).mean()
			else:
				loss = criterion(logits, y.float())
			loss.backward()"""

    new_loss_calc = """			if task_type == "multiclass":
				loss = criterion(logits, y.long())
			else:
				loss = criterion(logits, y.float())
			loss.backward()"""
    content = content.replace(old_loss_calc, new_loss_calc)

    # 4. Replace Targeted Validation
    old_targeted = """	azithromycin_smiles = resolve_smiles_from_pubchem("Azithromycin")
	haloperidol_smiles = resolve_smiles_from_pubchem("Haloperidol")
	if not azithromycin_smiles:
		azithromycin_smiles = TARGETED_QT_TEST_SMILES["Azithromycin"]
	if not haloperidol_smiles:
		haloperidol_smiles = TARGETED_QT_TEST_SMILES["Haloperidol"]
	print("\\nTargeted QT validation (Azithromycin + Haloperidol):")
	if azithromycin_smiles and haloperidol_smiles and idx_to_class:
		targeted = predict_pair_multiclass(
			model,
			device,
			azithromycin_smiles,
			haloperidol_smiles,
			args.max_nodes,
			idx_to_class,
		)"""

    new_targeted = """	citalopram_smiles = resolve_smiles_from_pubchem("Citalopram")
	amiodarone_smiles = resolve_smiles_from_pubchem("Amiodarone")
	if not citalopram_smiles:
		citalopram_smiles = "CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F"
	if not amiodarone_smiles:
		amiodarone_smiles = "CCCC1=C(C2=C(O1)C=CC(=C2)I)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I"
	print("\\nTargeted QT validation (Citalopram + Amiodarone):")
	if citalopram_smiles and amiodarone_smiles and idx_to_class:
		targeted = predict_pair_multiclass(
			model,
			device,
			citalopram_smiles,
			amiodarone_smiles,
			args.max_nodes,
			idx_to_class,
		)"""
    content = content.replace(old_targeted, new_targeted)

    # 5. Overwrite file
    with open("scripts/train_model.py", "w") as f:
        f.write(content)
        print("Modifications to scripts/train_model.py applied successfully.")

except Exception as e:
    traceback.print_exc()
