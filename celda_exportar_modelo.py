# ============================================================
# Agregar al FINAL del notebook (después del fine-tuning)
# ============================================================
from pathlib import Path

Path("model").mkdir(exist_ok=True)
torch.save(model.state_dict(), "model/modelo_hormigas_abejas.pth")
print("Modelo guardado en model/modelo_hormigas_abejas.pth")
print("Clases:", class_names)   # debe ser ['ants', 'bees']

# En Google Colab, descárgalo con:
# from google.colab import files
# files.download("model/modelo_hormigas_abejas.pth")
