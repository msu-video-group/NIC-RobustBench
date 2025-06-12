FROM debian:stable

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get upgrade -qqy \
 && apt-get install -qqy --no-install-recommends ffmpeg docker.io wget ca-certificates imagemagick
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip zip git dos2unix libgl1 libglib2.0-0
