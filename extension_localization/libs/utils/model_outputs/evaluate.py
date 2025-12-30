import pandas as pd
import numpy as np

# --- CONFIGURAZIONE ---
FILE_GT = 'gt.csv'
FILE_PREDS = 'preds.csv'
IOU_THRESH = 0.5  # La soglia standard (tIoU = 0.50)

def compute_iou(start_p, end_p, start_g, end_g):
    """Calcola Intersection over Union."""
    inter_start = max(start_p, start_g)
    inter_end = min(end_p, end_g)
    intersection = max(0, inter_end - inter_start)
    
    union = (end_p - start_p) + (end_g - start_g) - intersection
    if union <= 0: return 0
    return intersection / union

def main():
    print("--- CARICAMENTO E PREPARAZIONE ---")
    # Caricamento dati
    try:
        df_gt = pd.read_csv(FILE_GT, index_col=0)
        df_pred = pd.read_csv(FILE_PREDS, index_col=0)
    except:
        df_gt = pd.read_csv(FILE_GT)
        df_pred = pd.read_csv(FILE_PREDS)
        
    # Pulizia nomi colonne
    df_gt.columns = df_gt.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    print(f"Predizioni totali grezze: {len(df_pred)}")
    # IMPORTANTE: Forziamo tutte le label a 0 per renderlo "Class Agnostic"
    # Così il confronto avverrà solo sui tempi
    df_gt['label'] = 0
    df_pred['label'] = 0
    
    # Ordiniamo le predizioni globalmente per score decrescente (fondamentale per mAP)
    df_pred = df_pred.sort_values(by='score', ascending=False).reset_index(drop=True)

    print(f"GT Totali: {len(df_gt)}")
        # AGGIUNGI QUESTO BLOCCO DI CODICE PER VELOCIZZARE
    # Filtra via tutto ciò che ha una confidenza ridicola (sotto l'1% o il 5%)
    SOGLIA_TAGLIO = 0.05  # Prova 0.01 o 0.05
    df_pred = df_pred[df_pred['score'] > SOGLIA_TAGLIO].copy()
    
    print(f"Predizioni dopo il taglio (score > {SOGLIA_TAGLIO}): {len(df_pred)}")
    # --- CALCOLO mAP (Mean Average Precision) ---
    # Per calcolare la mAP dobbiamo scorrere le predizioni dalla più sicura alla meno sicura
    # e vedere se "colpiscono" un GT non ancora preso.
    
    tp = np.zeros(len(df_pred)) # True Positives array
    fp = np.zeros(len(df_pred)) # False Positives array
    
    # Teniamo traccia dei GT già "consumati" per ogni video per non contarli doppi
    # Struttura: { video_id: [True, False, False...] }
    gt_used_status = {}
    
    # Raggruppiamo i GT per video per accesso veloce
    gts_by_video = df_gt.groupby('video-id')
    
    for vid, group in gts_by_video:
        gt_used_status[vid] = np.zeros(len(group), dtype=bool)

    # Ciclo su tutte le predizioni (ordinate per score)
    for i, row_p in df_pred.iterrows():
        vid = row_p['video-id']
        
        # Se non ci sono GT per questo video, è sicuramente un False Positive
        if vid not in gt_used_status:
            fp[i] = 1
            continue
            
        # Recuperiamo i GT di questo video
        gt_intervals = gts_by_video.get_group(vid)[['t-start', 't-end']].values
        
        # Calcoliamo IoU con tutti i GT del video
        iou_max = -1
        gt_max_idx = -1
        
        for j, (start_g, end_g) in enumerate(gt_intervals):
            iou = compute_iou(row_p['t-start'], row_p['t-end'], start_g, end_g)
            if iou > iou_max:
                iou_max = iou
                gt_max_idx = j
        
        # Verifica se è un match valido
        if iou_max >= IOU_THRESH:
            # È un match, ma il GT è libero?
            if not gt_used_status[vid][gt_max_idx]:
                tp[i] = 1
                gt_used_status[vid][gt_max_idx] = True # Segna come preso
            else:
                fp[i] = 1 # GT già preso da una predizione con score più alto
        else:
            fp[i] = 1 # Nessuna sovrapposizione sufficiente

    # Calcolo curve Precision/Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    num_total_gts = len(df_gt)
    recalls = tp_cumsum / num_total_gts
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calcolo Average Precision (Area Under Curve approssimata)
    # Metodo standard COCO/Pascal VOC: smoothing della precisione
    ap = 0
    for t in np.arange(0, 1.1, 0.1): # 0, 0.1, ... 1.0
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0

    # --- CALCOLO RECALL @ K (Recall@1, Recall@5) ---
    # Questo si calcola video per video
    
    recall_at_1_list = []
    recall_at_5_list = []
    
    unique_videos = df_gt['video-id'].unique()
    
    for vid in unique_videos:
        current_gts = df_gt[df_gt['video-id'] == vid]
        num_gt_video = len(current_gts)
        if num_gt_video == 0: continue
            
        # Prendi le predizioni di questo video ordinate per score
        current_preds = df_pred[df_pred['video-id'] == vid] # Sono già ordinate globalmente, ma verifichiamo ordine locale
        
        # Calcolo R@1 (Top 1 predizione)
        top1_pred = current_preds.head(1)
        hits_1 = 0
        for _, row_p in top1_pred.iterrows():
            for _, row_g in current_gts.iterrows():
                if compute_iou(row_p['t-start'], row_p['t-end'], row_g['t-start'], row_g['t-end']) >= IOU_THRESH:
                    hits_1 += 1
                    break # Una predizione può hittare solo un GT qui semplifichiamo per R@K
        recall_at_1_list.append(min(hits_1 / num_gt_video, 1.0)) # Clip a 1

        # Calcolo R@5 (Top 5 predizioni)
        top5_preds = current_preds.head(5)
        # Qui dobbiamo fare attenzione: contare quanti GT distinti sono coperti dalle top 5
        covered_gts = set()
        for _, row_p in top5_preds.iterrows():
            for idx_g, row_g in current_gts.iterrows():
                if idx_g in covered_gts: continue
                if compute_iou(row_p['t-start'], row_p['t-end'], row_g['t-start'], row_g['t-end']) >= IOU_THRESH:
                    covered_gts.add(idx_g)
                    break # Questa pred ha trovato un GT, passa alla prossima pred
        
        recall_at_5_list.append(len(covered_gts) / num_gt_video)

    avg_recall_1 = np.mean(recall_at_1_list) * 100
    avg_recall_5 = np.mean(recall_at_5_list) * 100
    map_score = ap * 100

    print("\n" + "="*70)
    print(f"|tIoU = {IOU_THRESH:.2f}: mAP = {map_score:.2f} (%) Recall@1x = {avg_recall_1:.2f} (%) Recall@5x = {avg_recall_5:.2f} (%)")
    print(f"Average mAP: {map_score:.2f} (%)")
    print("="*70)

if __name__ == "__main__":
    main()