import yaml
import sys
import re

def main():
    if len(sys.argv) != 2:
        print('Usage: python parse_test_categories.py <yaml_file>')
        sys.exit(1)

    yaml_file = sys.argv[1]

    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Error loading YAML: {e}', file=sys.stderr)
        sys.exit(1)

    categories = config.get('test_categories', {})
    timeouts = config.get('execution_settings', {}).get('category_timeouts', {})

    print('# Generated CMake code for test categories')

    for category_name, category_info in categories.items():
        patterns = category_info.get('test_patterns', [])
        labels = category_info.get('labels', [])
        exclude = category_info.get('exclude', [])
        if exclude == None:
            exclude = []
        timeout = timeouts.get(category_name, 300)
        print(f'# Category: {category_name}')
        print(f'# Description: {category_info.get("description", "")}')
        patterns[:]=[x for x in patterns if x not in exclude]


        pattern_string=""
        for pattern in patterns:
          pattern_string = pattern_string+':'+pattern
        pattern_string='"'+pattern_string[1:]+'"'

        label_string = ""
        for label in labels:
          label_string = label_string+';'+label
        label_string = '"'+label_string[1:]+'"'
        print("add_test(")
        print(f'  NAME hipblas-{category_name}-suite')
        print(f'  COMMAND hipblas-test --gtest_filter={pattern_string}')
        print( "  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/staging")
        print(")")

        print(f"set_tests_properties(hipblas-{category_name}-suite PROPERTIES")
        print(f"  LABELS {label_string}")
        print(f"  TIMEOUT {timeout}")
        print(")")
        print()
if __name__ == '__main__':
    main()

