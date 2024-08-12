import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm

# from custom_OD_post_process_tools import get_all_predicted_detr_annotations_parallel_v1, obtener_datos_escalado_prediccion_detr_v1, group_detr_annotations_data_v1, filter_detr_annotations_data_v2, plot_predicted_annotations_data
# from custom_OD_prediction_tools import realizar_proceso_prediccion_detr_de_imagenes_spliteadas_v1

def guardar_datos_localmente(data, nombre_archivo):
    import pickle

    with open(nombre_archivo, 'wb') as archivo:
        pickle.dump(data, archivo)

def cargar_datos_localmente(nombre_archivo):
    import pickle
    with open(nombre_archivo, 'rb') as archivo:
        data = pickle.load(archivo)
    return data


def es_carpeta(ruta):
    return os.path.isdir(ruta)

def create_folder(ruta):
    import shutil

    # Remove the directory if it already exists
    if os.path.exists(ruta):
        # os.rmdir(ruta_completa_del_dataset)
        shutil.rmtree(ruta)

    # Create the directory
    if not os.path.exists(ruta):
        os.makedirs(ruta, exist_ok=True)


def get_folder_name_from_full_path(full_path):
    folder_name = os.path.basename(full_path)
    return folder_name

def cargar_imagen_to_bgr(ruta_nombre_archivo):
    import cv2
    return cv2.cvtColor(cv2.imread(ruta_nombre_archivo), cv2.COLOR_RGB2BGR)

def convertir_imagen_from_rgb_to_bgr(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

def convertir_imagen_from_bgr_to_rgb(imagen):
    import cv2
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

def plotear_mascara(imagen, image_name):
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(maskara, cmap='gray')
    ax.imshow(imagen)

    # Ajustar el diseño del gráfico y centrar la figura
    # plt.tight_layout()

    plt.title(f'Imagen: {image_name}')

    plt.show()
    

def copiar_archivo(archivo_original, ruta_destino):
    # Verificar si el archivo existe antes de copiarlo
    if os.path.exists(archivo_original):
        import shutil

        shutil.copy2(archivo_original, ruta_destino)
    else:
        print(f"NO existe el archivo_original: {archivo_original}")


from custom_pre_process_tools import read_and_correct_image_orientation, proceso_autorecortado_imagen, split_images

from secundary_extra_tools import custom_print
from secundary_extra_tools import get_current_date_formatted, time_difference_v2
from secundary_extra_tools import es_carpeta, listar_imagenes_en_ruta, crear_directorio, crear_archivo_txt
from secundary_extra_tools import cargar_imagen_to_rgb, get_file_name_and_extension_from_full_path, convertir_imagen_from_rgb_to_bgr, guardar_imagen

from custom_OD_post_process_tools import get_all_predicted_OD_yolov8_annotations_parallel_v1, obtener_datos_escalado_prediccion_OD_v1, filter_OD_NMS_annotations_data_v1
from custom_OD_prediction_tools import realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v4

ruta_base = f"{os.getcwd()}"
ruta_salida = f"{os.getcwd()}/predictions_epoch12_mi_raspberry"

# ruta_images = f"{os.getcwd()}/test_640x640_v1"
ruta_images = f"{os.getcwd()}/some_full_images/cel_jrt"

image_id_start = 0

ruta_base = ruta_base.replace("\\","/")
ruta_images = ruta_images.replace("\\","/")

s_width = 640
s_height = 640
overlap = 0.75
ruta_checkpoint_bbox = f"{os.getcwd()}/best_weight_epoch12_tflite_yolov8n".replace("\\","/")
# weight_file_name = "best"
weight_file_name = "best_weight_ds_official_epch250_float32_v2"
second_weight_file_name = "best_weight_recorte_region_interes_official_epch12_float32_v1"

# algorithm_name = "yolov8n"
# algorithm_name = "yolov8n_tflite"
algorithm_name = "yolov8n_tflite_OD" # for confusion matrix of OD
classes_names = ["objeto_interes"]
custom_model_device = "cpu"
batch_size = 4
custom_checkpoint_name = get_folder_name_from_full_path(f"{ruta_checkpoint_bbox}")
custom_checkpoint_name = custom_checkpoint_name.replace("-", "_")

# model_path = f"{ruta_checkpoint_bbox}/{weight_file_name}.pt"
model_path = f"{ruta_checkpoint_bbox}/{weight_file_name}.tflite"
model_path_auto_recorte_imagen = f"{ruta_checkpoint_bbox}/{second_weight_file_name}.tflite"

if es_carpeta(ruta_images):

    print(f"")

    images_path_list = listar_imagenes_en_ruta(ruta_images)

    # output_folder_name = f"output_OD_yolov8_tflite_epoch12"
    custom_images_folder_name = get_folder_name_from_full_path(f"{ruta_images}")

    ruta_salida = f"{ruta_salida}/{algorithm_name}/{custom_images_folder_name}"
    crear_directorio(ruta_salida)

    nuevo_id_imagen = image_id_start
    object_count_per_image_list = []

    initialTime = datetime.now()
    formattedTime = initialTime.strftime("%d/%m/%Y, %H:%M:%S")

    total_images_a_recorrer = len(images_path_list)
    
    from ultralytics import YOLO
    model = YOLO(model_path)
    model2 = YOLO(model_path_auto_recorte_imagen)
    
    
    
    # for index, image_file in enumerate(images_path_list):

    for i_idx_image, image_file in tqdm(enumerate(images_path_list), desc="Progreso prediccion de imagenes", total=len(images_path_list)):

        confidence_threshold_configs_dict = {}
        iou_thresholds = [0.30] # list of iou_thresholds of current confidence_threshold
        iou_thresholds_conteo_total_results = [0] # list of counting in different iou_thresholds of current confidence_threshold

        for confidence_threshold in [0.60]:
            confidence_threshold_configs_dict[confidence_threshold] = {
                'iou_thresholds': iou_thresholds,
                'conteos': iou_thresholds_conteo_total_results
            }

        list_of_images = []

        # anotaciones de segmentacion (coordenadas flatten)
        instances_annotations_list = []

        image_file_name, image_file_extension = get_file_name_and_extension_from_full_path(image_file)

        full_image_loaded = read_and_correct_image_orientation(f"{ruta_images}/{image_file}")
        full_image_loaded = proceso_autorecortado_imagen(full_image_loaded, model2)

        height_full_image_loaded, width_full_image_loaded = full_image_loaded.shape[:2]
        
        copy_full_image_loaded = convertir_imagen_from_rgb_to_bgr(full_image_loaded.copy())
        ruta_salida_imagen = f"{ruta_salida}/{image_file_name}.jpg"
        
        guardar_imagen(copy_full_image_loaded, f"{ruta_salida}/{image_file_name}.jpg")
        # copiar_archivo(f"{ruta_images}/{image_file}",f"{ruta_salida}/{image_file}")

        initialTime_resume_over_all_images = None
        initialTime_resume_over_all_images = datetime.now()

        ######################################### OBTENER IMAGENES SPLITEADAS #######################################
        split_images_dict, image_splits_keys, image_splits_list = split_images(full_image_loaded, split_width=s_width, split_height=s_height, overlap=overlap)

        imgsz = s_height
        imgsz = s_width
        
        # custom_print(f"", f"[{image_file_name}]", has_len=False)
        print(f"\n")
        print(f"{image_file_name}")
        
        
        for confidence_threshold, data in confidence_threshold_configs_dict.items():
            # print(f'Confidence Threshold: {confidence_threshold}')
            iou_thresholds = data['iou_thresholds']
            conteos = data['conteos']
            
            # confidence_threshold_configs_results = "imageName,ConfidenceThreshold,nmsThreshold,totalPredicciones\n"
            confidence_threshold_configs_results = "imageName,ConfidenceThreshold,nmsThreshold,totalPredicciones,tiempoTomado\n"

            nombre_txt_file = f"{image_file_name}.jpg_{confidence_threshold}_conf_predictions_epoch12_mi_maquina"
            ruta_txt_archivo = f"{ruta_salida}/{nombre_txt_file}.txt"
            
            ######################################### PREDECIR LOS SPLITS DE LA IMAGEN Y ESCALAR LAS PREDICCIONES DE LOS SPLITS HACIA LAS DIMENSIONES DE LA IMAGEN ORIGINAL #######################################
            all_results, _ = realizar_proceso_prediccion_yolov8_de_imagenes_spliteadas_v4(model, imgsz, image_splits_list, custom_model_device, confidence_threshold)
            
            ######################################### PREDECIR LOS SPLITS DE LA IMAGEN Y ESCALAR LAS PREDICCIONES DE LOS SPLITS HACIA LAS DIMENSIONES DE LA IMAGEN ORIGINAL #######################################
            scores, centroids, bboxs, category_ids, keys = get_all_predicted_OD_yolov8_annotations_parallel_v1(all_results, image_splits_keys)
            scores, centroids, bboxs, category_ids, keys = obtener_datos_escalado_prediccion_OD_v1((scores, centroids, bboxs, category_ids, keys), full_image_loaded)
            
            for index, (iou_threshold, conteo) in enumerate(zip(iou_thresholds, conteos)):
                
                nuevo_id_annotation = 0
                
                selected_scores, selected_centroids, selected_bboxs, selected_category_ids, selected_keys = filter_OD_NMS_annotations_data_v1(scores, centroids, bboxs, category_ids, keys, iou_threshold)

                finalTime_resume_over_all_images = None
                finalTime_resume_over_all_images = datetime.now()

                custom_print(f"conf_thr={confidence_threshold}, iou_thr={iou_threshold}, conteo={len(selected_bboxs)}, tiempo_tomado={time_difference_v2(initialTime_resume_over_all_images, finalTime_resume_over_all_images)}", f"", has_len=False)

                cont_filtered_predicted_annotations_data = 0
                
                # confidence_threshold_configs_results += f"{image_file_name}.jpg,{confidence_threshold},{iou_threshold},{len(selected_bboxs)}\n"
                confidence_threshold_configs_results += f"{image_file_name}.jpg,{confidence_threshold},{iou_threshold},{len(selected_bboxs)},{time_difference_v2(initialTime_resume_over_all_images, finalTime_resume_over_all_images)}\n"
                
                for i_idx in range(len(selected_bboxs)):
                        
                    score, centroid, bbox, category_id = selected_scores[i_idx], selected_centroids[i_idx], selected_bboxs[i_idx], selected_category_ids[i_idx]

                    x_min = int(bbox[0])
                    y_min = int(bbox[1])
                    x_max = int(bbox[2])
                    y_max = int(bbox[3])

                    instance_annotation_info = {
                        "score": score,
                        "centroid": centroid,
                        "segmentation": [],
                        "area": None,
                        "iscrowd": 0,
                        "image_id": i_idx_image,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "category_id": 1,
                        "id": nuevo_id_annotation
                    }

                    cont_filtered_predicted_annotations_data += 1

                    instances_annotations_list.append(instance_annotation_info)
                    nuevo_id_annotation += 1

                images_info = {
                    "conteo_total": len(selected_bboxs),
                    "tiempo_tomado": time_difference_v2(initialTime_resume_over_all_images, finalTime_resume_over_all_images),
                    "file_name": f"{image_file_name}.jpg",
                    "height": height_full_image_loaded,
                    "width": width_full_image_loaded,
                    "date_captured": get_current_date_formatted(),
                    "id": i_idx_image
                }

                if not any(get_data_image["file_name"] == images_info["file_name"] for get_data_image in list_of_images):
                    list_of_images.append(images_info)

                list_of_categories = []
                
                categories_info = {
                    "id": 1,
                    "name": "objeto_interes",
                    "supercategory": "objeto_interes"
                }
                
                if not any(get_data_categorie["name"] == categories_info["name"] for get_data_categorie in list_of_categories):
                    list_of_categories.append(categories_info)

                coco_json = {
                    "images": list_of_images,
                    "annotations": instances_annotations_list,
                    "categories": list_of_categories
                }


                salida_json_coco_instances_anotaciones = f"{ruta_salida}/{image_file_name}.jpg_{confidence_threshold}_conf_thr_{iou_threshold}_iou_thr_epoch12_{algorithm_name}_mi_maquina.json"
                salida_json_coco_instances_anotaciones_salida = f"{ruta_salida}/{image_file_name}.jpg_{confidence_threshold}_conf_thr_{iou_threshold}_iou_thr_epoch12_{algorithm_name}_mi_maquina_salida.json"

                # guardar anotaciones de instancias de coco en un archivo json
                with open(salida_json_coco_instances_anotaciones, 'w') as f:
                    json.dump(coco_json, f)
                
                with open(salida_json_coco_instances_anotaciones_salida, 'w') as f:
                    json.dump(instances_annotations_list, f)

            crear_archivo_txt(ruta_txt_archivo, confidence_threshold_configs_results)

        


        ###################### GUARDAR RESULTADOS OBTENIDOS SOBRE LAS IMAGENES ######################

    print(f" \n \n")

    # print(f"[LAST] salida_json_coco_instances_anotaciones: {salida_json_coco_instances_anotaciones}")
    print(f' La Prediccion sobre las imagenes Terminó (╯°□°)╯ ')
    

