ARG METRIC_IMAGE=None
FROM ${METRIC_IMAGE} as base

#COPY --from=vg-code.gml-team.ru:4567/framework/attacks/datasets/coco:25e_shu /data /

RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install av
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install PyWavelets
RUN pip3 install IQA_pytorch
RUN pip3 install frozendict
RUN pip3 install lpips
RUN pip3 install torchmetrics
RUN pip3 install pytorch-wavelets
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install pyiqa
RUN pip3 install pytorch-msssim
RUN pip3 install kornia

COPY ./read_dataset.py /read_dataset.py
COPY ./metrics.py /metrics.py
# COPY ./bounds.json /bounds.json
COPY ./evaluate.py /evaluate.py
COPY ./fgsm_evaluate.py /fgsm_evaluate.py
# COPY ./uap_evaluate.py /uap_evaluate.py
# COPY ./cnn_evaluate.py /cnn_evaluate.py
COPY ./color_transforms_255.py /color_transforms_255.py
COPY ./attack_presets_codecs.json /attack_presets_codecs.json
# COPY ./attack_presets.json /attack_presets.json
COPY ./defended_model.py /defended_model.py
COPY ./defence_scoring_methods.py /defence_scoring_methods.py
COPY ./picked_presets_all_v1.csv ./picked_presets_all_v1.csv 
COPY ./codec_losses.py ./codec_losses.py
COPY ./codec_scoring_methods.py ./codec_scoring_methods.py
COPY ./traditional_reference_codec.py /traditional_reference_codec.py

COPY run.py /run.py

