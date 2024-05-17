from flask import Flask, request, render_template,jsonify
import os
from PIL import Image,ImageEnhance
import io
import base64
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import pytesseract
import easyocr
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import glob
import csv
import uuid



CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/drag')
def drag():
    return render_template('drag.html')
@app.route('/start_live_video',methods=['POST'])
def live():
    def save_results(text,region,csv_filename,folder_path):
        img_name='{}.jpg'.format(uuid.uuid1())
        cv2.imwrite(os.path.join(folder_path,img_name),region)
        with open(csv_filename,mode='a',newline='')as f:
            csv_writer=csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([img_name,text])
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=10,
                    min_score_thresh=.4,
                    agnostic_mode=False)
        try:
            text,region=ocr_it(image_np_with_detections,detections,detection_threshold,region_threshold)
            save_results(text,region,'live_detected_result.csv','Detection_images')
        except:
            pass
            
        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status':'OK'})







@app.route('/upload',methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'no image found'  
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    # IMAGE_PATH ="D:\images\plate.png"


    # img = cv2.imread(IMAGE_PATH)
    file=request.files['image']
    file.save('uploads/'+file.filename)
    # file.save('test_data/'+file.filename+".jpg")
    img=cv2.imread("uploads/"+file.filename)
    img=np.array(img)
    img_path="uploads/"+file.filename
    if img.shape[1]<500 or img.shape[0]<300:
        folder_path="D:\\Minor Projects\\anpr_demo\\test_data"
        files1=glob.glob(f'{folder_path}/*')
        for file1 in files1:
            if(os.path.exists(file1)):
                os.remove(file1)
        folder_path="D:\\Minor Projects\\anpr_demo\\result"
        files1=glob.glob(f'{folder_path}/*')
        for file in files1:
            if(os.path.exists(file1)):
                os.remove(file1)
        cv2.imwrite(os.path.join("test_data","orignal.jpg"),img)
        img_path="result/res_0000.png"
        os.system("python main.py --mode test_only --LR_path test_data --generator_path pretrained_models/SRGAN.pt")
    img=cv2.imread(img_path)
    image_np = np.array(img)
    # plt.imshow(image_np)
    # plt.show()

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.4,
                agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()


    detection_threshold=0.4
    image=image_np_with_detections
    scores=list(filter(lambda x:x>detection_threshold,detections['detection_scores']))
    boxes=detections['detection_boxes'][:len(scores)]
    classes=detections['detection_classes'][:len(scores)]

    width=image.shape[1]
    height=image.shape[0]
    ocr=""
    tess=[]
    region=""
    ocr_result=""
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Bunty\AppData\Local\Programs\Tesseract-OCR\tesseract'
    for idx,box in enumerate(boxes):
        roi=box*[height,width,height,width]
        region=image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        if region.shape[1]<100 or region.shape[0]<20:
            folder_path="D:\\Minor Projects\\anpr_demo\\test_data"
            files1=glob.glob(f'{folder_path}/*')
            for file1 in files1:
                if(os.path.exists(file1)):
                    os.remove(file1)
            folder_path="D:\\Minor Projects\\anpr_demo\\result"
            files1=glob.glob(f'{folder_path}/*')
            for file in files1:
                if(os.path.exists(file1)):
                    os.remove(file1)
            cv2.imwrite(os.path.join("test_data","orignal.jpg"),region)
            img_path="result/res_0000.png"
            os.system("python main.py --mode test_only --LR_path test_data --generator_path pretrained_models/SRGAN.pt")
            region=cv2.imread(img_path)
            region=np.array(region)
        #     cv2.imwrite(os.path.join("test_data","orignal.jpg"),region)
        #     os.system("python main.py --mode test_only --LR_path test_data --generator_path pretrained_models/SRGAN.pt")
        #     region=cv2.imread(os.path.join("result","res_0000.png"))
        #     sr = cv2.dnn_superres.DnnSuperResImpl_create()
        #     path = "LapSRN_x8.pb"
        #     sr.readModel(path)
        #     sr.setModel("lapsrn",8)
        #     # path = "EDSR_x4.pb"
        #     # sr.readModel(path)
        #     # sr.setModel("edsr",4)
        #     region=sr.upsample(region)
            # cv2.imwrite(os.path.join("uploads","orignal.jpg"),region)
            # region=Image.open(os.path.join("uploads","orignal.jpg"))
            # new_bri=0.7
            # new_con=1.8
            # new_sharp=9.0
            # new_clr=0.0
            # region=ImageEnhance.Brightness(region).enhance(new_bri)
            # region=ImageEnhance.Contrast(region).enhance(new_con)
            # region=ImageEnhance.Sharpness(region).enhance(new_sharp)
            # region= np.array(region)
            #enhance Brightness
            # curr_bri=ImageEnhance.Brightness(region)
            # region=curr_bri.enhance(1.5)
            # print("called")
        # cv2.imwrite(os.path.join("uploads","orignal.jpg"),region)
        # region=Image.open(os.path.join("uploads","orignal.jpg"))
        # new_bri=0.7
        # new_con=1.5
        # new_sharp=15.0
        # new_clr=0.0
        # region=ImageEnhance.Brightness(region).enhance(new_bri)
        # region=ImageEnhance.Contrast(region).enhance(new_con)
        # region=ImageEnhance.Sharpness(region).enhance(new_sharp)
        # region= np.array(region)
        reader=easyocr.Reader(['en'])
        ocr_result=reader.readtext(region)
        cv2.imwrite("Detected_plate//res.jpg",region)
        ocr_result
        text = str(pytesseract.image_to_string(region))
        # print(ocr_result)
        # plt.imshow(cv2.cvtColor(region,cv2.COLOR_BGR2RGB))
        # print(text)
        tess.append(text)
        print(text)
        # cv2.imwrite(os.path.join("D:\images","detect.jpg"),region)
    region_threshold=0.4
    def filter_text(region,ocr_result,region_threshold):
        rectangle_size=region.shape[0]*region.shape[1]
        plate=[]
        for result in ocr_result:
            length=np.sum(np.subtract(result[0][1],result[0][0]))
            height=np.sum(np.subtract(result[0][2],result[0][1]))
            if length*height/rectangle_size>region_threshold:
                plate.append(result[1])
        return plate   
    
      
    ocr=filter_text(region,ocr_result,region_threshold)
    for x in range(len(ocr)):
        if ocr[x].isalpha():
            ocr[x]=ocr[x].upper()

    buffered = io.BytesIO()
    Image.fromarray(region).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')   
    return jsonify({'region':img_str,'easy': ocr,'tess':tess})
if __name__ =="__main__":
    app.run(debug=True)







