import urllib.request
import datetime
import json
import pandas as pd
import xml.etree.ElementTree as ET

# 인코딩된 서비스 키
ServiceKey = "UKkAe8jvrkN%2FT9mQ2kldttyWkMq2zV5%2Bi8RNss%2F8TvkCb95RyVfgHgxMq4MUt4BcyXNoz5y46QkuGeKclDfLSA%3D%3D"

def getRequestUrl(url):
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print(f"[{datetime.datetime.now()}] Url Request Success")
            return response.read().decode('utf-8')
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] Error for URL : {url}")
        return None

def getHeatWaveData(yyyymm, type="xml"):
    service_url = "http://apis.data.go.kr/1741000/DaysHeatWavesMajorCitiesYear/getDaysHeatWavesMajorCitiesYearList"
    
    parameters = f"?_type={type}&serviceKey={ServiceKey}"
    parameters += f"&bas_yy={yyyymm}"
    
    url = service_url + parameters
    
    retData = getRequestUrl(url)
    
    if retData is None:
        return None
    else:
        try:
            if type == "json":
                return json.loads(retData)
            elif type == "xml":
                return retData
        except json.JSONDecodeError:
            print(f"Error decoding JSON for year {yyyymm}: {retData}")
            return None

def parseXML(xmlData):
    root = ET.fromstring(xmlData)
    rows = []
    for row in root.findall('row'):
        bas_yy = row.find('bas_yy').text
        lseoul = row.find('lseoul').text
        lgangneung = row.find('lgangneung').text
        ldaejeon = row.find('ldaejeon').text
        ldaegu = row.find('ldaegu').text
        lgwangju = row.find('lgwangju').text
        lbusan = row.find('lbusan').text
        anat_dd_avg = row.find('anat_dd_avg').text
        rows.append({
            'bas_yy': bas_yy,
            'lseoul': lseoul,
            'lgangneung': lgangneung,
            'ldaejeon': ldaejeon,
            'ldaegu': ldaegu,
            'lgwangju': lgwangju,
            'lbusan': lbusan,
            'anat_dd_avg': anat_dd_avg
        })
    return rows

def getHeatWaveStats(start_year, end_year, type="xml"):
    jsonResult = []
    result = []
    
    for year in range(start_year, end_year + 1):
        xmlData = getHeatWaveData(year, type)
        
        if xmlData:
            items = parseXML(xmlData)
            for item in items:
                jsonResult.append(item)
                result.append([
                    item.get('bas_yy', ''),
                    item.get('lseoul', 0),
                    item.get('lgangneung', 0),
                    item.get('ldaejeon', 0),
                    item.get('ldaegu', 0),
                    item.get('lgwangju', 0),
                    item.get('lbusan', 0),
                    item.get('anat_dd_avg', 0.0)
                ])
    
    return jsonResult, result

def main():
    start_year = 2007
    end_year = 2022
    
    jsonResult, result = getHeatWaveStats(start_year, end_year)
    
    # JSON 파일 저장
    with open(f'./heatwave_data_{start_year}_{end_year}.json', 'w', encoding='utf8') as outfile:
        jsonFile = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
        outfile.write(jsonFile)
    
    # CSV 파일 저장
    columns = ["기준년도", "서울", "강릉", "대전", "대구", "광주", "부산", "전국일평균"]
    result_df = pd.DataFrame(result, columns=columns)
    result_df.to_csv(f'./heatwave_data_{start_year}_{end_year}.csv', index=False, encoding='cp949')

if __name__ == '__main__':
    main()
