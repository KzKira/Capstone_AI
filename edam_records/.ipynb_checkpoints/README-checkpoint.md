# Microplastic Object Detection Dataset
---

## Image
이미지의 높이 및 너비는 224px이며, 자료형은 uint8입니다.

## Label
label의 shape은 (16,16,6,7) 입니다.

Bounding box는 (x,y,w,h,confidence,isFragment,isFiber)로 구성되어있습니다.

## TFRecord를 어떻게 사용해야하나요?
how_to_parse.ipynb 파일을 참고해주세요!