# AutoCorrection
## Requirements
You will need <a href="https://git-scm.com/book/en/v2/Getting-Started-Installing-Git" target="_blank">Git</a> and <a href="https://docs.docker.com/desktop/" target="_blank">Docker Desktop</a> installed
## Installation
Open a command prompt to git clone this repo
```bash
git clone https://github.com/LGrAr/AutoCorrection.git
```
change directory to the cloned repo
```bash
cd AutoCorrection
```
build the docker image
```bash
docker build -t <image_name> .
```
run the docker image
```
docker run -p 5000:5000 <image_name>
```
click <a href="http://localhost:5000/" target="_blank">http://localhost:5000/</a> or <a href="http://0.0.0.0:5000/" target="_blank">http://0.0.0.0:5000/</a> to open a web browser and access the app
