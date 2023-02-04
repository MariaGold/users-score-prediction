FROM python:3.6

#MAINTAINER Maria Goldshtein

RUN apt-get update -y && apt-get install -y nodejs && apt-get install -y npm && npm install nodemon -g

RUN mkdir -p /app

COPY req.txt ./

RUN pip install --upgrade pip && pip install -r req.txt

WORKDIR /app

EXPOSE 8080

CMD [ "nodemon", "--exec", "python", "./__init__.py"]