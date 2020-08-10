import requests
import json

def sent_message(mess):
	_data = {"message":mess}
	r = requests.post("http://test-local-284817.appspot.com/notification/sendmobile",data=json.dumps(_data))
	print(r.text)

# if __name__=='__main__':
# 	sent_message("Cảnh báo khẩn cấp. Người thân của bạn gặp sự cố !!!")