#EXAMPLE INVOCATIONS :

#author@VirtualTasteTeam

#VirtualTasteapiscript

#Python version: 3.6 and above

#Customized example : query server for all model data, based on a smiles-string, and output to out.csv.
#PLEASE NOTE : As seen below, do add quotation marks if you include a SMILES string, since otherwise the = will be mis-interpreted. Likewise,use quotes around the whole query if split drugnames (two words) occur within
#python simple_api.py -t smiles -m ALL_MODELS -o out.csv "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"
#python simple_api.py aspirin,sertraline

import requests;    #Server interaction
import time;        #Timers for waiting time estimation
import argparse;    #Command line switch handling
import json;        #Structured data formatting and transfer
import sys;
import urllib;

#List of all computationally intensive models. Either manually pick from this list, or specify to use ALL_MODELS directly
ALL_MODELS = ["Bitter","Sweet"]

#PROGRAM LOGIC
#---------------------------------
parser=argparse.ArgumentParser(description="Query the VirtualTaste API")
parser.add_argument("-t","--mtype", help="Default: name. specify 'name' (pubchem search using the compound name) or 'smiles' (canonical smiles) as your input data type",type=str,choices=['name','smiles'],default="name")
parser.add_argument("-m","--models",help="Default : Bitter,Sweet. specify models and data points to compute. Options are (to be separated by ,): Bitter,Sweet and the additional model sour. You can use all additional models using ALL_MODELS, but be mindful it may incur high calculation times",type=str,default="Bitter,Sweet")
parser.add_argument("searchterms", help="The actual strings to search for (comma-separated). May be pubchem name (default) or SMILES (use -t smiles). ",type=str)
parser.add_argument("-o","--outfile",help="Default : results.csv. specify where to output your retrieved data (csv format)",type=str,default="results.csv")
parser.add_argument("-r","--retry",help="Default : 5. Retry limit to attempt to retrieve data. Increase this in case of large, unpredictable queries",type=int,default=5)
parser.add_argument("-q","--quiet",help="Suppress all non-error output from the script",action="store_true")
#parser.add_argument("-q","--quiet",help="Suppress all non-error output from the script",action="store_true")
args=parser.parse_args()

input_type=args.mtype;
models=args.models.split(',')
searchterms=args.searchterms.split(',')
outfile=args.outfile;
retry_limit=args.retry;
quiet=args.quiet;

try:
    data_file=open(outfile,"w")
except IOError:
    print ("Could not open specified outfile")
    sys.exit();

task_id_list=[]

def log(msg):
  if (not quiet):
    print (msg)

def request_data(inputs):
    log("Enqueing request "+inputs+", with models :")

    if "ALL_MODELS" in models:
        models.remove("ALL_MODELS")
        models.extend(ALL_MODELS)

    log(models)
    r=requests.post("https://insilico-cyp.charite.de/VirtualTaste/src/api_enqueue_new.php",data={'input_type':input_type,'input':inputs,'requested_data':json.dumps(models)}) #encode array, the rest are single strings
    if (r.status_code==200): #Data response

        #Data query response. Add response to our task_id_list
        log("Recieved qualified response with id")
        #log(r.text)
        task_id_list.append(r.text) #See if this works.

        #Set up wait time before next query
        if 'Retry-After' in r.headers:
          wait_time=int(r.headers['Retry-After'])+1 #Wait depending on how long the server thinks it needs
        else:
          wait_time=0 #Something went wrong with transmission, just wait 10s by default

        log("Waiting for "+str(wait_time)+" s till next request")

        time.sleep(wait_time)

    elif (r.status_code==429): #Too many requests. Slow down/wait
            log("Server responds : Too many requests. Slowing down query slightly")
            if 'Retry-After' in r.headers:
                wait_time=r.headers['Retry-After']+1 #Wait depending on how long the server thinks it needs
            else:
                wait_time=0 #Something went wrong with transmission, just wait 10s by default

            log("Waiting for "+wait_time+" s till next request")
            time.sleep(wait_time)
            #WOULD HAVE TO RE-SUBMIT?

    elif (r.status_code==403): #Daily quota exceeded. Aborting operation for now.
            data_file.write(",Daily Quota Exceeded"); #Input as well
            print ("Daily Quota Exceeded. Terminating process.")
            exit(0);

    else: #Server gone away or different issues, outputting here
        print ("ERROR : Server issue")
        print (r.status_code,r.reason)


for inputs in searchterms:
    request_data(inputs)

#Once done, retrieve data from the server. Repeat queries every 30s until all data has been retrieved. Append to an outfile if given
log ("All queries have been enqueued. Starting result retrieval...")
time.sleep(2);

result_objects=[]

while task_id_list: #As long as we still have elements to get :
  response_list=[]
  first_response=1
  for task_id in task_id_list:
    #log("Asking for "+task_id)
    #print(task_id)
    r=requests.post("https://insilico-cyp.charite.de/VirtualTaste/src/api_retrieve.php",data={'id':task_id})
    if (r.status_code==200): #Data response
      if (r.text==""):
        print ("Warning : Empty response")
      else:
        response_list.append(task_id)
        #f = open("//insilico-cyp.charite.de/VirtualTaste/csv/"+task_id+"_result.csv", "r")
        response=urllib.request.urlopen("https://insilico-cyp.charite.de/VirtualTaste/csv/"+task_id+"_result.csv")
        the_page = str(response.read())
        the_page2 = the_page.replace("b'", "")
        the_page3 = the_page2.replace("'", "")
        the_page3 = the_page3.replace("\\r\\n", "\n")
        data_file.write(the_page3)

		#print(f.read())
        #log("Recieved queue id: "+r.text+"\n")
        #for line in r.text.splitlines():
         #print (line)
        #log("Recieved queue id2: "+task_id+"\n")
        #result_objects.append(json.loads(r.text))
        #log("Recieved queue id: "+task_id+"\n")
    elif (r.status_code==404): #Not found, not computed or finished yet. Do nothing
      if (first_response):
        #log("No response yet. Likely cause: computation unfinished (retrying...)")
        print (r.text)
        first_response=0
    else: #Other codes are not permitted
      print ("Unexpected return from server")
      print (r.status_code,r.reason)
      sys.exit();

    #response_list.append(r.text) #Add to found list. Now   #Output result to console or file

  task_id_list=[item for item in task_id_list if item not in response_list] #Remove all found id's
  if (task_id_list):
    #log("Some queries still pending. Retrying in 1s")
    sys.exit();
	#time.sleep(0); #Wait 1s before another run if there's still work to do


#data_file.write(json.dumps(result_objects))
#data_file.write(r.text)
data_file.close()

print ("Completed all operations. Your results are in "+outfile)