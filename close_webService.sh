#!/bin/sh
configFile="./paramConfig.conf"
 
function ReadINIfile()  
{   
	Key=$1
	Section=$2
  	Configfile=$3
	ReadINI=`awk -F '=' '/\['$Section'\]/{a=1}a==1&&$1~/'$Key'/{print $2;exit}' $Configfile`  
 	echo "$ReadINI"  
} 
 
pid=`ReadINIfile "pid" "webserviceParam" "$configFile"`

echo 'close pids:'
echo $pid

function str_split()
{
	OLD_IFS="$IFS"
	IFS=','
	for each in $1; do
		echo $each
	done
	unset each
	IFS="$OLD_IFS"

}
#str_split "$subportlist"
str_list=`str_split "$pid"`


for i in $str_list; do
	kill -9 $i
done
unset i

python operationConfig.py clrearPid

echo "O.K" 
