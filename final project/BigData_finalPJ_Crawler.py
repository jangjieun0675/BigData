import urllib.request
import datetime
import json
import pandas as pd
import xml.etree.ElementTree as ET

# 인코딩된 서비스 키
SERVICE_KEY = "UKkAe8jvrkN%2FT9mQ2kldttyWkMq2zV5%2Bi8RNss%2F8TvkCb95RyVfgHgxMq4MUt4BcyXNoz5y46QkuGeKclDfLSA%3D%3D"

def get_request_url(url):
    """URL에서 데이터를 가져오는 함수"""
    req = urllib.request.Request(url)
    try:
        response = urllib.request.urlopen(req)
        if response.getcode() == 200:
            print(f"[{datetime.datetime.now()}] URL 요청 성공")
            return response.read().decode('utf-8')
        else:
            print(f"HTTP 에러 코드: {response.getcode()}")
            return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        print(f"HTTP 에러 코드: {e.code}")
        return e.read().decode('utf-8')  # 에러 메시지 반환
    except Exception as e:
        print(e)
        print(f"[{datetime.datetime.now()}] URL 요청 오류: {url}")
    return None

def get_forest_fire_data(search_st_dt, search_ed_dt, num_of_rows=1000, page_no=1, response_type="xml"):
    """특정 기간 동안의 산불 데이터를 가져오는 함수"""
    service_url = "http://apis.data.go.kr/1400000/forestStusService/getfirestatsservice"
    parameters = (
        f"?_type={response_type}&serviceKey={SERVICE_KEY}"
        f"&searchStDt={search_st_dt}&searchEdDt={search_ed_dt}"
        f"&numOfRows={num_of_rows}&pageNo={page_no}"
    )
    url = service_url + parameters
    print(f"Request URL: {url}")  # 디버깅을 위해 URL 출력
    
    return get_request_url(url)

def parse_xml(xml_data):
    """XML 데이터를 파싱하여 구조화된 형식으로 변환하는 함수"""
    root = ET.fromstring(xml_data)
    rows = []
    for item in root.findall('.//item'):
        row = {
            'damagearea': item.find('damagearea').text if item.find('damagearea') is not None else '',
            'endday': item.find('endday').text if item.find('endday') is not None else '',
            'endmonth': item.find('endmonth').text if item.find('endmonth') is not None else '',
            'endtime': item.find('endtime').text if item.find('endtime') is not None else '',
            'endyear': item.find('endyear').text if item.find('endyear') is not None else '',
            'firecause': item.find('firecause').text if item.find('firecause') is not None else '',
            'locbunji': item.find('locbunji').text if item.find('locbunji') is not None else '',
            'locdong': item.find('locdong').text if item.find('locdong') is not None else '',
            'locgungu': item.find('locgungu').text if item.find('locgungu') is not None else '',
            'locmenu': item.find('locmenu').text if item.find('locmenu') is not None else '',
            'locsi': item.find('locsi').text if item.find('locsi') is not None else '',
            'startday': item.find('startday').text if item.find('startday') is not None else '',
            'startdayofweek': item.find('startdayofweek').text if item.find('startdayofweek') is not None else '',
            'startmonth': item.find('startmonth').text if item.find('startmonth') is not None else '',
            'starttime': item.find('starttime').text if item.find('starttime') is not None else '',
            'startyear': item.find('startyear').text if item.find('startyear') is not None else ''
        }
        rows.append(row)
    return rows

def get_forest_fire_stats(search_st_dt, search_ed_dt, response_type="xml"):
    """특정 기간 동안의 산불 데이터를 조회하고 파싱하는 함수"""
    json_result = []
    result = []
    page_no = 1
    while True:
        xml_data = get_forest_fire_data(search_st_dt, search_ed_dt, num_of_rows=1000, page_no=page_no, response_type=response_type)
        if xml_data:
            if "<resultCode>00</resultCode>" not in xml_data:
                print(f"에러 응답: {xml_data}")
                break
            items = parse_xml(xml_data)
            if not items:
                break
            for item in items:
                json_result.append(item)
                result.append([
                    item.get('damagearea', ''),
                    item.get('endday', ''),
                    item.get('endmonth', ''),
                    item.get('endtime', ''),
                    item.get('endyear', ''),
                    item.get('firecause', ''),
                    item.get('locbunji', ''),
                    item.get('locdong', ''),
                    item.get('locgungu', ''),
                    item.get('locmenu', ''),
                    item.get('locsi', ''),
                    item.get('startday', ''),
                    item.get('startdayofweek', ''),
                    item.get('startmonth', ''),
                    item.get('starttime', ''),
                    item.get('startyear', '')
                ])
            page_no += 1
        else:
            break
    
    return json_result, result

def main():
    search_st_dt = '20160101'
    search_ed_dt = '20231231'
    
    json_result, result = get_forest_fire_stats(search_st_dt, search_ed_dt)
    
    if json_result and result:
        # JSON 파일 저장
        with open(f'./forest_fire_data_{search_st_dt}_{search_ed_dt}.json', 'w', encoding='utf8') as outfile:
            json_file = json.dumps(json_result, indent=4, sort_keys=True, ensure_ascii=False)
            outfile.write(json_file)
        
        # CSV 파일 저장
        columns = [
            "피해면적", "진화종료일", "진화종료월", "진화종료일시", "진화종료연도", 
            "발생원인", "발생장소_지번", "발생장소_동리", "발생장소_시군구", "발생장소_읍면", 
            "발생장소_시도", "발생일", "발생요일", "발생월", "발생시간", "발생연도"
        ]
        result_df = pd.DataFrame(result, columns=columns)
        result_df.to_csv(f'./forest_fire_data_{search_st_dt}_{search_ed_dt}.csv', index=False, encoding='cp949')

if __name__ == '__main__':
    main()
