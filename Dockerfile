# Use an official Python runtime as the base image
FROM ubuntu:20.04
RUN  pip install opencv-python-headless
RUN pip install matplotlib
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install tesseract
RUN pip install pytesseract
