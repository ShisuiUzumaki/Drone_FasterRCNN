from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import cv2

config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
#cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "visdrone_model_0360000.pth"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

# load image and then run prediction
# # image = cv2.imread('visdrone_test_img_0000001_02999_d_0000005.jpg')
# paths = glob(r'../../../input/aerialdataset/*.jpg')
# if not os.path.exists('/kaggle/working/predictions/images'):
#     os.makedirs('/kaggle/working/predictions/images')
# if not os.path.exists('/kaggle/working/predictions/bbox'):
#     os.makedirs('/kaggle/working/predictions/bbox')
# sp_ann = '/kaggle/working/predictions/bbox'
# for path in paths:
#     img = cv2.imread(path)
#     file_name = path.split('/')[-1]
#     predictions = coco_demo.run_on_opencv_image(img, file_name, path=sp_ann)
#     print(f'/kaggle/working/predictions/images/{file_name.split('.')[0]}_DET.jpg')
#     cv2.imwrite(f'/kaggle/working/predictions/images/{file_name.split('.')[0]}_DET.jpg', predictions)
image = cv2.imread('visdrone_test_img_0000001_02999_d_0000005.jpg')
predictions = coco_demo.run_on_opencv_image(image)
#cv2.imwrite('drone_res.jpg', predictions)
cv2.imshow('Predictions', predictions)
cv2.waitKey(0)
