FROM python:3.9
COPY api.py requirements.txt ./
ADD https://github.com/33tm/Parrot/releases/download/model/data.csv out/data.csv
ADD https://github.com/33tm/Parrot/releases/download/model/model.pt out/model.pt
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "api.py"]