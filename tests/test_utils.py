# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"Test utilities."

import xml.etree.ElementTree as ET

import html5lib

html_parser = html5lib.HTMLParser()


def compare_html_string(a: str, b: str) -> bool:
  a_normalized = ET.tostring(html_parser.parse(a))
  b_normalized = ET.tostring(html_parser.parse(b))
  return a_normalized == b_normalized
