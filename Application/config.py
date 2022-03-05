from dotenv import dotenv_values
var = dotenv_values(".env")
SUBSCRIPTION_KEY = var['SUBSCRIPTION_KEY']
LOCATION = var['LOCATION']