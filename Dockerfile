FROM ubuntu
MAINTAINER pranav
WORKDIR /home/ec2-user
RUN apt-get update -y
RUN apt install python3
RUN apt-get install python3-pip
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
#ENTRYPOINT["spark-submit"]
#ENTRYPOINT ["python" "./PredictionModel.py"]
#ENTRYPOINT ["python", "PredictionModel.py"]
ENTRYPOINT ["python3", "home/ec2-user/PredictionModel.py"]
