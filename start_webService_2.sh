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
 
subportlist=`ReadINIfile "subportlist" "webserviceParam" "$configFile"`
port=`ReadINIfile "port" "webserviceParam" "$configFile"`

echo $subportlist
echo $port

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
str_list=`str_split "$subportlist"`


echo | nohup python billTypeWebService_v2.py $port &
echo | nohup python billTypeWebService_v2_sub.py 10002 &
echo | nohup python billTypeWebService_v2_sub.py 10003 &
echo | nohup python billTypeWebService_v2_sub.py 10004 &
echo | nohup python billTypeWebService_v2_sub.py 10005 &


#for i in $str_list; do
#	echo | nohup python billTypeWebService_v2_sub.py $i &
#done
#unset i

echo "O.K" 
