 #!/bin/bash                                                                                                                                                                  
                                                                                                                                                                              
 # Check if the CSV file is provided as an argument                                                                                                                           
 if [ "$#" -ne 1 ]; then                                                                                                                                                      
     echo "Usage: $0 <csv_file>"                                                                                                                                              
     exit 1                                                                                                                                                                   
 fi                                                                                                                                                                           
                                                                                                                                                                              
 CSV_FILE="$1"                                                                                                                                                                
                                                                                                                                                                              
 # Check if the file exists                                                                                                                                                   
 if [ ! -f "$CSV_FILE" ]; then                                                                                                                                                
     echo "File not found: $CSV_FILE"                                                                                                                                         
     exit 1                                                                                                                                                                   
 fi                                                                                                                                                                           
                                                                                                                                                                              
 # Read the header to find the positions of 'ra' and 'dec' columns                                                                                                            
 HEADER=$(head -n 1 "$CSV_FILE")                                                                                                                                              
 IFS=',' read -r -a COLUMNS <<< "$HEADER"                                                                                                                                     
                                                                                                                                                                              
 RA_INDEX=-1                                                                                                                                                                  
 DEC_INDEX=-1                                                                                                                                                                 
                                                                                                                                                                              
 for i in "${!COLUMNS[@]}"; do                                                                                                                                                
     if [ "${COLUMNS[$i]}" == "ra" ]; then                                                                                                                                    
         RA_INDEX=$i                                                                                                                                                          
     elif [ "${COLUMNS[$i]}" == "dec" ]; then                                                                                                                                 
         DEC_INDEX=$i                                                                                                                                                         
     fi                                                                                                                                                                       
 done                                                                                                                                                                         
                                                                                                                                                                              
 # Check if both 'ra' and 'dec' columns were found                                                                                                                            
 if [ "$RA_INDEX" -eq -1 ] || [ "$DEC_INDEX" -eq -1 ]; then                                                                                                                   
     echo "Error: 'ra' and/or 'dec' columns not found in the CSV file."                                                                                                       
     exit 1                                                                                                                                                                   
 fi                                                                                                                                                                           
                                                                                                                                                                              
 # Iterate over each line in the CSV file (excluding the header)                                                                                                              
 tail -n +2 "$CSV_FILE" | while IFS=',' read -r -a LINE; do                                                                                                                   
     RA="${LINE[$RA_INDEX]}"                                                                                                                                                  
     DEC="${LINE[$DEC_INDEX]}"                                                                                                                                                
                                                                                                                                                                              
     # Run the Python script with the extracted RA and DEC                                                                                                                    
     python lightcurvequery.py "$RA" "$DEC"                                                                                                                                   
 done                                                                                                                                                                         
               
