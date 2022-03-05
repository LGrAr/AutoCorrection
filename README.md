# AutoCorrection
A web app to automatically correct students answers for online plateforms.
It allows:
  * Professors to add their responses to questions to get reference answers
  * similarity indexes between students and teachers answers
  * binary classification to auto correct students answers
  * Professors to translate french exercices to other languages
  * Manual correction
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
or run the docker image with your azure subscription key/location if you have one, to be able to use the translation feature
```
docker run -e SUBSCRIPTION_KEY=<your_subscription key> -e LOCATION=<your_azure_location> -p 5000:5000 <image_name>
```
click <a href="http://localhost:5000/" target="_blank">http://localhost:5000/</a> or <a href="http://0.0.0.0:5000/" target="_blank">http://0.0.0.0:5000/</a> to open a web browser and access the app
