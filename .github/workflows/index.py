

# incase zip file---> S3 
# S3Bucket: my-lambda-code-bucket
# S3Key: path/to/lambda-code.zip


import json
import boto3

def lambda_handler(event, context):
    glue = boto3.client("glue")
    response = glue.start_workflow_run(Name="yelpworkflow")
    print(response)
    return {
        'statusCode' : 200,
        'body' : json.dumps("Hello from AWS Lambda!")
    }

