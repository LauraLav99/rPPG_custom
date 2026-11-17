import argparse,os, time, sys, json,shutil, subprocess
from datetime import datetime, timedelta, timezone
import cv2, mediapipe as mp, numpy as np


from rPPG_custom.rppg_facedetection_utils import build_face_detector,compute_rois,draw_rois
from rPPG_custom.rppg_utils import (return_camera_files,trainlab_rppg_parser,extract_rgb_means,run_algo,plot_stack,
                                    write_outputs)
from rPPG_custom.rppg_methods import fuse_signals

# ---------------------------------------------------------------------------
# ROI = poligoni costruiti, frame per frame, sui landmark FaceMesh (478 punti).
# Per ogni regione si prende l'involucro convesso (convex hull) dei landmark
# elencati e lo si rimpicciolisce verso il centro (--roi-shrink) per stare
# lontani da capelli e sfondo. La media RGB e' calcolata solo sui pixel dentro
# il poligono. Occhi, sopracciglia, naso e bocca restano fuori per costruzione
# (verificato: nessuno dei loro landmark cade dentro i poligoni).
# Per cambiare una regione basta modificare l'elenco di indici.
#   cheek_l = lato SINISTRO dell'immagine (guancia destra del soggetto)
#   cheek_r = la sua riflessa (indici ottenuti per simmetria dal modello canonico)
# ---------------------------------------------------------------------------
ROI_POLYGONS = {
    "forehead": [103, 67, 109, 10, 338, 297, 332,  68, 104, 69, 108, 151, 337, 299, 333, 298],
    "cheek_l":  [117, 229, 230, 120, 100, 142, 203, 206, 216, 207, 187, 147, 123, 116],
    "cheek_r":  [346, 449, 450, 349, 329, 371, 423, 426, 436, 427, 411, 376, 352, 345],
}
ROI_PRESETS = {
    "forehead":        ["forehead"],
    "cheeks":          ["cheek_l", "cheek_r"],
    "forehead+cheeks": ["forehead", "cheek_l", "cheek_r"],
}
ROI_COLORS = {"forehead": (0, 0, 255), "cheek_l": (0, 255, 0), "cheek_r": (255, 0, 0)}   # BGR
MIN_ROI_PIXELS = 100    # sotto questa area (px) il frame viene scartato
model=os.path.join(os.getcwd(),"face_landmarker.task")

ALGO_NAMES = {"pos": "POS_WANG", "chrom": "CHROM_DEHAAN", "green": "GREEN",
              "multi": "MULTI(POS+CHROM+GREEN)", "fused": "FUSED"}
ROI=[ROI_PRESETS,ROI_POLYGONS]

def main():
    p = argparse.ArgumentParser()
    args=trainlab_rppg_parser(p,ROI_PRESETS)
    # create the output filder in the input foldrr
    output_folder=os.path.join(args.input_folder,args.folder)
    os.makedirs(output_folder,exist_ok=True)
    #1. Find the camera files and json files in the input folder
    camerafiles=return_camera_files(args)
    rec_tag=None

    for ff in camerafiles:
        file=camerafiles[ff]
        input=os.path.join(args.input_folder,file)
        fname = os.path.splitext(os.path.basename(input))[0]
        jsonfile=fname+'.json'
        with open(os.path.join(args.input_folder,jsonfile), "r") as file:
            metadata = json.load(file)
        rec_start=(datetime.strptime(["fasi"]["rec_start"], '%Y-%m-%d %H:%M:%S.%f')).time()

        cap=cv2.VideoCapture(input)
        #replace if not with try and catch
        if not cap.isOpened():
            raise RuntimeError("Impossibile aprire la sorgente video")
        fs_meta=cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"FPS da metadati: {fs_meta:.3f}")
        if args.verbose:
            print(f"Sorgente: '{input}' | {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frame")
            print("Timestamp: CAP_PROP_POS_MSEC (OpenCV, letto dopo ogni cap.read)")
        if args.output is None:
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"ppg_{args.mode}_{fname}_{ts}"
        # add a crop here beforeto build a facedeector? less memory consuming
        face_detector=build_face_detector(model)
        if args.verbose:
            print(f"ROI: {', '.join(ROI_PRESETS[args.roi])}")

        rgb_buf, ts_buf,timestamp_buf = [], [], []
        last_rois = None
        n_missed = 0  # frame senza volto rilevato (si riusano le ultime ROI)
        n_seen = 0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

            now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ts= rec_start + now
            rgb_val = None
            n_seen += 1
            detected = True
            res = face_detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            detected = bool(res.face_landmarks)
            if detected:
                last_rois = compute_rois(res.face_landmarks[0], ROI_PRESETS[args.roi],
                                         frame.shape, args.roi_shrink)
            else:
                n_missed += 1  # si riusano le ultime ROI note
            if last_rois:
                rgb_val = extract_rgb_means(frame, last_rois)
            if rgb_val is not None:
                rgb_buf.append(rgb_val); ts_buf.append(now);timestamp_buf.append(ts)
            if not args.no_preview:
                if args.mode=="face" and last_rois:
                    draw_rois(frame, last_rois, detected)
                cv2.putText(frame,f"t={now:.1f}s n={len(ts_buf)}",
                            (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                cv2.imshow("rPPG preview (non salvata)", frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break

        finally:
            if face_detector: face_detector.close()
            cap.release(); cv2.destroyAllWindows()

        if args.mode == "face" and n_missed:
            print(f"[WARN] Volto non rilevato in {n_missed}/{n_seen} frame ({100 * n_missed / n_seen:.1f}%): "
                  f"in quei frame sono state riusate le ultime ROI valide")
        if len(ts_buf) < 30:
            print("Troppi pochi campioni.");return

        rgb_arr = np.stack(rgb_buf)  # (N, 3)
        ts_arr = np.array(ts_buf)  # (N,)
        N = len(ts_arr)
        dur = ts_arr[-1] - ts_arr[0]

        # fs: metadati per file (timing esatto), misurata per webcam (più fedele al reale)
        fs = (N - 1) / dur
        print(f"fs usata: {fs:.3f} Hz")
# If signals are long, we conuld introduce a windowing of the signal. POS use a detrend function thta uses
        #invert matrices-> high memory consumption slow
        metainfo=[]
        metainfo[0]=n_missed
        metainfo[1]=n_seen
        metainfo[2] = fs
        metainfo[3] = fs_meta
        metainfo[4]=rec_tag
        metainfo[5]=dur
        metainfo[6]=rec_start
        if args.algo == "all":
            # Stessi dati RGB per tutti: il video e' letto una volta sola.
            runs = {}
            for name in ("pos", "chrom", "green"):
                print(f"{ALGO_NAMES[name]}...")
                try:
                    sig, wu, vn = run_algo(name, rgb_arr, fs)
                except ValueError as e:
                    print(f"[WARN] {ALGO_NAMES[name]} saltato: {e}")
                    continue
                sig = sig.astype(np.float64)
                sig[vn:] = np.nan
                #modificare write output in order to have also the timestamp sig
                # (out_name, signals,rgb_arr, main_name, warmup_n,rec_start,ALGO_NAMES,ROI,ts_arr,args,metainfo, info=None, algo_label=None):
                res = write_outputs(f"{output_folder}_{name}", {name: sig}, name,rgb_arr,
                                    wu,ALGO_NAMES,ROI,ts_arr,args,metainfo)
                runs[name] = {"signal": sig, "warmup_n": wu, "hr": res["hr"]}
            if len(runs) < 2:
                print("Meno di due algoritmi utilizzabili: nessun confronto.")
            else: # comparsion only if there are more tnan 2 algorithms
                start = max(r["warmup_n"] for r in runs.values())
                # saltiamo parte confronto per ora
                # try:
                #     comp = compare_signals({k: r["signal"] for k, r in runs.items()}, fs, start)
                # except ValueError as e:
                #     comp = None
                #     print(f"[WARN] confronto non eseguito: {e}")
                # if comp:
                #     print(f"\nConfronto (porzione comune dopo warm-up: {comp['common_samples']} campioni, "
                #           f"consenso {comp['consensus_hr_bpm']:.1f} bpm)")
                #     print(f"{'algoritmo':<14}{'HR [bpm]':>10}{'SNR [dB]':>10}")
                #     for k in runs:
                #         h = runs[k]["hr"]
                #         print(
                #             f"{ALGO_NAMES[k]:<14}{(f'{h:.1f}' if h else '-'):>10}{comp['snr_db_at_consensus'][k]:>10.1f}")
                #     print("|correlazione| a coppie: " +
                #           "  ".join(f"{k}={v:.2f}" for k, v in comp["abs_correlation"].items()))
                #     comp["hr_bpm"] = {k: r["hr"] for k, r in runs.items()}
                #     cj = os.path.join(args.folder, args.output + "_compare.json")
                #     with open(cj, "w") as f:
                #         json.dump(comp, f, indent=2)
                #     print(f"Confronto JSON: {cj}")
                # modify plot stack in order to return only the comaprison plot
                # if args.plot:
                #         labels = {k: f"{ALGO_NAMES[k]}  HR {r['hr']:.1f}" if r["hr"] else ALGO_NAMES[k]
                #                   for k, r in runs.items()}
                #
                #         plot_stack(args.output + "_compare.png", {k: r["signal"] for k, r in runs.items()},
                #                    start, title="Confronto algoritmi (stesso video, stesse ROI)", labels=labels)
        elif args.algo == "multi":
            signals, wu_list = {}, []
            for name in ("pos", "chrom", "green"):
                try:
                    sig, wu, vn = run_algo(name, rgb_arr, fs)
                except ValueError as e:
                    print(f"[WARN] {ALGO_NAMES[name]} saltato: {e}")
                    continue
                sig = sig.astype(np.float64)
                sig[vn:] = np.nan
                signals[name] = sig
                wu_list.append(wu)
            if len(signals) < 2:
                raise RuntimeError("--algo multi richiede almeno 2 algoritmi utilizzabili")
            fused, info = fuse_signals(signals, fs, win_sec=args.fusion_win, mode=args.fusion)
            signals["fused"] = fused

            res = write_outputs(output_folder,signals,"fused", rgb_arr,
                                max(wu_list), ALGO_NAMES["MULTI"], ROI, ts_arr, args, metainfo)
        else:
            print(f"{ALGO_NAMES[args.algo]}...")
            sig, wu, vn = run_algo(args.algo, rgb_arr, fs)
            sig = sig.astype(np.float64)
            sig[vn:] = np.nan
            write_outputs(output_folder, {args.algo: sig}, args.algo, rgb_arr,
                          wu,ALGO_NAMES,ROI,ts_arr,args,metainfo)
        # signals[nome][n] <-> ts_arr[n]  (allineamento 1:1)
















if __name__ == "__main__":
    main()

