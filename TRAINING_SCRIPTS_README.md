# 🚀 Script per Training Automatizzati

## 📁 File creati:

1. **`run_all_trainings.sh`** - Script principale (2 training paralleli)
2. **`run_all_trainings_fast.sh`** - Versione veloce (3 training paralleli)
3. **`monitor_trainings.sh`** - Monitoraggio in tempo reale

---

## 🎯 Cosa fanno gli script

Eseguono **12 training** con tutte le combinazioni:

- **Variants**: MLP, RNN, Transformer
- **Backbones**: omnivore, perception
- **Configurazioni**: 
  - **OLD**: pos_weight=2.5, lr=1e-3, weight_decay=1e-3
  - **NEW**: pos_weight=1.6, lr=5e-4, weight_decay=5e-3
- **Parametri fissi**: split=step, modality=video, epochs=50

---

## 🚀 Come usare

### Opzione 1: Versione standard (2 paralleli, ~6-8 ore)

```bash
cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
./run_all_trainings.sh
```

### Opzione 2: Versione veloce (3 paralleli, ~5-6 ore)

```bash
cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
./run_all_trainings_fast.sh
```

### Monitorare i progressi

```bash
# Esegui in un altro terminale
./monitor_trainings.sh

# Oppure guarda un log specifico
tail -f nohup_OLD_MLP_omnivore.out
tail -f nohup_NEW_Transformer_perception.out
```

---

## 📊 Output

### Log files

I log saranno salvati come:
```
nohup_OLD_MLP_omnivore.out
nohup_OLD_MLP_perception.out
nohup_OLD_RNN_omnivore.out
...
nohup_NEW_Transformer_perception.out
```

### Checkpoints

Salvati in:
```
./checkpoints/error_recognition/
├── MLP/
│   ├── omnivore/
│   └── perception/
├── RNN/
│   ├── omnivore/
│   └── perception/
└── Transformer/
    ├── omnivore/
    └── perception/
```

### WandB

Ogni run sarà tracciata su WandB con nome:
```
{VARIANT}_{BACKBONE}_step_video_e50_bs128_lr{LR}
```

---

## 🛑 Fermare i training

### Graceful stop (consigliato)
```bash
pkill -INT -f "python train_er.py"
```

### Force kill
```bash
pkill -9 -f "python train_er.py"
```

---

## ⏱️ Stima tempi

### Con 2 training paralleli (run_all_trainings.sh):
- 6 gruppi di 2 training
- ~1 ora per training (50 epoche)
- **Totale: ~6-8 ore**

### Con 3 training paralleli (run_all_trainings_fast.sh):
- 4 gruppi di 3 training
- ~1 ora per training (50 epoche)
- **Totale: ~5-6 ore**

---

## 📋 Ordine di esecuzione

1. **Prima fase** (OLD config, pos_weight=2.5):
   - MLP omnivore + MLP perception (paralleli)
   - RNN omnivore + RNN perception (paralleli)
   - Transformer omnivore + Transformer perception (paralleli)

2. **Seconda fase** (NEW config, pos_weight=1.6):
   - MLP omnivore + MLP perception (paralleli)
   - RNN omnivore + RNN perception (paralleli)
   - Transformer omnivore + Transformer perception (paralleli)

---

## ⚠️ Note importanti

- Gli script modificano automaticamente `base.py` per cambiare il `pos_weight`
- Dopo il completamento, `pos_weight` viene ripristinato a 1.5
- Lo script usa `conda activate aml` e rimuove pyenv dal PATH
- Ogni training genera un file di log separato
- I training vengono eseguiti in background con `nohup`

---

## 🐛 Troubleshooting

### Se un training crasha:
1. Controlla il log: `tail -100 nohup_[CONFIG]_[VARIANT]_[BACKBONE].out`
2. Verifica che conda sia attivo e Python sia 3.11
3. Rilancia manualmente quel specifico training

### Per rilanciare un singolo training:
```bash
export PATH=$(echo $PATH | tr ":" "\n" | grep -v pyenv | tr "\n" ":" | sed "s/:$//")
cd /home/pierpaolosorbellini/AML/aml-2025-mistake-detection
conda activate aml

python train_er.py \
    --variant MLP \
    --backbone omnivore \
    --split step \
    --modality video \
    --num_epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-3
```

(Ricorda di modificare `pos_weight` in `base.py` se necessario!)

---

## ✅ Verifica completamento

Dopo il completamento, verifica:

```bash
# Conta i checkpoint creati
find checkpoints/error_recognition/ -name "*.pt" | wc -l

# Mostra gli ultimi checkpoint per ogni modello
find checkpoints/error_recognition/ -name "*_epoch_50.pt"
```

Dovresti avere checkpoint per tutte le 12 combinazioni!


