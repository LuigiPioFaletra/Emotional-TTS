import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

def augment_noise(X, y, noise_factor=0.01, num_copies=1):
    augmented = []
    augmented_labels = []
    for _ in range(num_copies):
        noise = np.random.normal(0, noise_factor, X.shape)
        augmented.append(X + noise)
        augmented_labels.append(y)
    X_aug = np.vstack(augmented)
    y_aug = np.concatenate(augmented_labels)
    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])

np.random.seed(42)

# ==========================
# Ignora warning di convergenza
# ==========================
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================
# Caricamento dati
# ==========================
embeddings = np.load("embeddings.npy")  # shape (N, 1024)
df = pd.read_csv("metadata.csv", sep="\t")

labels = df['emotion'].values
splits = df['split'].values

X_train = embeddings[splits == 'train']
y_train = labels[splits == 'train']

X_val = embeddings[splits == 'validation']
y_val = labels[splits == 'validation']

X_test = embeddings[splits == 'test']
y_test = labels[splits == 'test']

# ==========================
# Encoding etichette
# ==========================
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
y_test_enc = le.transform(y_test)

# ==========================
# Scaling delle embeddings
# ==========================
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ==========================
# Addestramento manuale con warm_start
# ==========================
epochs = 50
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0003,
    alpha=0.002,
    batch_size=128,
    max_iter=1,
    warm_start=True,
    n_iter_no_change=5,
    random_state=42
)

# ==========================
# Variabili per tracciare il miglior modello
# ==========================
best_val_acc = 0.0
best_model = None
best_epoch = 0

train_acc, val_acc = [], []
X_train_final, y_train_final = augment_noise(X_train, y_train_enc, noise_factor=0.02, num_copies=3)
base_weights = compute_sample_weight(class_weight="balanced", y=y_train_final)

for epoch in tqdm(range(epochs), desc="Training MLP"):
    X_train_epoch, y_train_epoch, w_epoch = shuffle(X_train_final, y_train_final, base_weights, random_state=epoch)
    mlp.fit(X_train_epoch, y_train_epoch, sample_weight=w_epoch)
    train_acc.append(mlp.score(X_train_epoch, y_train_epoch))
    val_acc.append(mlp.score(X_val, y_val_enc))
    
    # Seleziona e salva il miglior modello in base alla validation accuracy
    if val_acc[-1] > best_val_acc:
        best_val_acc = val_acc[-1]
        best_model = copy.deepcopy(mlp)
        best_epoch = epoch
        tqdm.write(f"Epoch {epoch+1}/{epochs} - Train acc: {train_acc[-1]:.4f} - Val acc: {val_acc[-1]:.4f} (BEST)")
    else:
        tqdm.write(f"Epoch {epoch+1}/{epochs} - Train acc: {train_acc[-1]:.4f} - Val acc: {val_acc[-1]:.4f}")

# Usa il miglior modello trovato
mlp = best_model
print(f"\nModello selezionato: epoca {best_epoch+1} con validation accuracy {best_val_acc:.4f}")

# ==========================
# Grafici
# ==========================
plt.figure(figsize=(10,4))
plt.plot(mlp.loss_curve_, label="Training Loss")
plt.title("Andamento della Loss durante il training")
plt.xlabel("Iterazioni interne")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title("Accuratezza su Train e Validation")
plt.xlabel("Epoca")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# Valutazione finale su validation
# ==========================
y_val_pred = mlp.predict(X_val)
print("\n=== VALIDATION ===")
print("Validation accuracy:", accuracy_score(y_val_enc, y_val_pred))
print(classification_report(y_val_enc, y_val_pred, target_names=le.classes_, zero_division=0))

disp_val = ConfusionMatrixDisplay.from_predictions(
    y_val_enc,
    y_val_pred,
    display_labels=le.classes_,
    cmap='Blues'
)
plt.title("Validation Confusion Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================
# Test finale
# ==========================
y_test_pred = mlp.predict(X_test)
print("\n=== TEST ===")
print("Test accuracy:", accuracy_score(y_test_enc, y_test_pred))
print(classification_report(y_test_enc, y_test_pred, target_names=le.classes_, zero_division=0))

disp_test = ConfusionMatrixDisplay.from_predictions(
    y_test_enc,
    y_test_pred,
    display_labels=le.classes_,
    cmap='Blues'
)
plt.title("Test Confusion Matrix")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================
# Salvataggio del modello
# ==========================
joblib.dump(mlp, "mlp_embeddings_model.joblib")
print("\nModello salvato come 'mlp_embeddings_model.joblib'")
print(f"Modello dalla epoca {best_epoch+1} con validation accuracy {best_val_acc:.4f}")
