"""
-------------------------------------------
confusion_class.py

Define the class used in confusion_main.
-------------------------------------------
"""
# using library
import os
import datetime
from xml.dom import minidom


class ScoringResult:
	type_str: str
	start_mills: int
	
	def __init__(self, type_str: str, start_mills: int):
		self.type_str = type_str
		self.start_mills = start_mills
	
	def to_list(self):
		return [self.type_str, self.start_mills]
	
	def from_rml(source_file: str) -> list:
		xml = minidom.parse(source_file)
		stages = xml.getElementsByTagName('Stage')
		start_time = xml.getElementsByTagName("RecordingStart")[0]
		during_time = xml.getElementsByTagName("Duration")[0]
		start_time = start_time.firstChild.data
		start_time = start_time.replace('T', ' ')
		start_time_object = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
		during_time = during_time.firstChild.data
		during_time_int = int(during_time)
		finish_time_object = start_time_object + during_time_int * datetime.timedelta(seconds=1)
		result = []
		for s in stages:
			t = s.attributes['Type'].value
			start = s.attributes['Start'].value
			delta_time = int(start)
			if delta_time != 0:
				during_epoc = int((delta_time - past_delta_time) / 30)
				
				for i in range(during_epoc):
					if i != 0:
						input_time = past_delta_time + i * 30
						measuring_time = start_time_object + input_time * datetime.timedelta(seconds=1)
						result.append(ScoringResult(str(measuring_time), past_t))
			
			measuring_time = start_time_object + delta_time * datetime.timedelta(seconds=1)
			result.append(ScoringResult(str(measuring_time), t))
			
			past_measuring_time = measuring_time
			past_t = t
			past_delta_time = delta_time
		
		if finish_time_object > past_measuring_time:
			num_epoc_fill = int((during_time_int - past_delta_time) / 30)
			for j in range(num_epoc_fill + 1):
				if j != 0:
					input_time = past_delta_time + j * 30
					measuring_time = start_time_object + input_time * datetime.timedelta(seconds=1)
					result.append(ScoringResult(str(measuring_time), past_t))
		
		return result


def to_list(element: ScoringResult) -> list:
	return [element.type_str, element.start_mills]