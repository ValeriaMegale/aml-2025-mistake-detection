# Refactoring Summary

## Completato ✅

### Struttura Finale
```
aml-2025-mistake-detection/
├── core/                           # Step 2: Baselines
├── dataloader/
├── annotations/
├── extension/                      # Extension: Task Verification
│   ├── substep1_step_localization/
│   ├── substep2_task_verification/
│   ├── substep3_task_graph_matching/
│   └── substep4_gnn_classification/
├── results/
│   ├── baselines/
│   └── extension/
├── scripts/                        # Script utility
├── docs/                           # Documentazione
└── REFACTORING_PLAN.md
```

### Modifiche Effettuate

1. ✅ **Fase 1**: Aggiunto `*.out` e `nohup*.out` a `.gitignore`
2. ✅ **Fase 2**: Consolidate directory extension in `extension/`
3. ✅ **Fase 3**: Script spostati in `scripts/`
4. ✅ **Fase 4**: Docs spostati in `docs/`
5. ✅ **Fase 5**: Risultati organizzati in `results/baselines/` e `results/extension/`
6. ✅ **Fase 8**: Aggiornati path e import (parzialmente)

### File Aggiornati

- `scripts/monitor_status.py` - Path aggiornati
- `REPORT.md` - Path aggiornati
- Tutti i file Python in `extension/` - Import e path aggiornati
- Tutti i file Markdown in `extension/` - Path aggiornati
- Script shell in `scripts/` - Path aggiornati

### Note Importanti

- **er_annotations/**: Mantenuto perché usato da `CaptainCookStepDataset.py`
- **extension_localization/**: Mantenuto temporaneamente (verificare se ancora necessario)
- **checkpoints_selected/**: Da verificare se necessario

### Prossimi Passi

1. Testare che tutti gli script funzionino con i nuovi path
2. Verificare directory non usate (extension_localization, checkpoints_selected)
3. Aggiornare README.md con struttura finale
4. Fare commit del refactoring
