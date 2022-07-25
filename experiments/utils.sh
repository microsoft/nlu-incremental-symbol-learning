# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#!/usr/bin/env bash


#######################################
# cecho prints a text using custom color
# Arguments:
#   -c or --color define the color for the print. See the array colors for the available options.
#   -n or --noline directs the system not to print a new line after the content.
#    Last argument is the message to be printed.
# Returns:
#   None
#######################################
function cecho () {

    declare -a COLORS;
    COLORS=(\
        ['black']='\E[0;47m'\
        ['red']='\E[0;31m'\
        ['green']='\E[0;32m'\
        ['yellow']='\E[0;33m'\
        ['blue']='\E[0;34m'\
        ['magenta']='\E[0;35m'\
        ['cyan']='\E[0;36m'\
        ['white']='\E[0;37m'\
        ['default']='\E[0;00m'\
    );

    local DEFAULT_MESSAGE="No message passed.";
    local DEFAULT_COLOR="default";
    local DEFAULT_NEW_LINE=true;

    while [[ $# -gt 1 ]];
    do
        arg="$1";
        case $arg in
            -c|--color)
                COLOR="$2";
                shift;
            ;;
            -n|--noline)
                NEW_LINE=false;
            ;;
            *)
                # unknown option
            ;;
        esac
        shift;
    done

    MESSAGE=${1:-$DEFAULT_MESSAGE};   # Defaults to default message.
    COLOR=${COLOR:-$DEFAULT_COLOR};   # Defaults to default color, if not specified.
    NEW_LINE=${NEW_LINE:-$DEFAULT_NEW_LINE};

    echo -en "${COLORS[$COLOR]}";
    echo -en "$MESSAGE";
    if [ "$NEW_LINE" = true ] ; then
        echo;
    fi
    echo -en "${COLORS[$DEFAULT_COLOR]}";
    tty -s && tput sgr0; #  Reset text attributes to normal without clearing screen.

    return;

}


function warning () {

    cecho -c 'yellow' "$@";
}


function error () {

    cecho -c 'red' "$@";

}


function information () {

    cecho -c 'blue' "$@";
}


function log_error() {

    cecho -n -c 'red' 'ERROR '
    printf "$@\t`date`\n"
}


function log_warn() {

    cecho -n -c 'yellow' 'WARN  '
    printf "$@\t`date`\n"

}


function log_info() {

    cecho -n -c 'cyan' 'INFO  '
    printf "$@\t`date`\n"

}


#######################################
# log_on_failure logs message on failure
# Arguments:
#   First positional argument is the return code
#   Second positional argument is the error message string
# Returns:
#   1 on failure and 0 on success
#######################################
function log_on_failure() {

    if [ $1 -ne 0 ]; then

        log_error "$2"

        return 1

    else

        return 0

    fi

}


#######################################
# terminate terminates the program with SUCCESS/FAILURE message
# Arguments:
#   First positional argument is the return code
#######################################
function terminate() {

    if [ $1 -ne 0 ]; then

        cecho -c 'red' '\nFAILED!\n'
        exit 1

    else

        cecho -c 'green' '\nSUCCEEDED!\n'
        exit 0

    fi

}


#######################################
# is_json_empty checks if json file is empty
# Arguments:
#   First positional argument is the json file path
# Returns:
#   1 on failure and 0 on success
#######################################
function is_json_empty() {

    if [[ $# -lt 1 ]]; then
        log_error "function is_json_empty needs an argument"
        exit 1
    fi

    __JCONTENTS=$(jq -r 'to_entries[] | "\(.key)=\(.value)"' $1)

    if [ -z "${__JCONTENTS}" ]; then

        log_warn "Empty json file $1"
        return 1

    else

        return 0

    fi

}


#######################################
# is_valid_json checks if the path points to a valid json
# Arguments:
#   First positional argument is the json file path
# Returns:
#   1 on failure and 0 on success
#######################################
function is_valid_json() {

    if [[ $# -lt 1 ]]; then
        log_error "function is_valid_json needs an argument"
        exit 1
    fi

    __JCONTENTS=$(jq -r 'to_entries[] | "\(.key)=\(.value)"' $1)
    RET_CODE=$?
    log_on_failure "$RET_CODE" "Invalid json file $1"

}


#######################################
# json_to_bash_variables converts and sources the flat json object into shell variables
# Arguments:
#   First positional argument is the json file path
# Returns:
#   1 on failure and 0 on success
#######################################
function json_to_bash_variables() {

    if [[ $# -lt 1 ]]; then
        log_error "function json_to_bash_variables needs an argument"
        exit 1
    fi

    {

        # Check if json is valid
        is_valid_json $1 &&

        # Check & fail if json is empty
        is_json_empty $1 &&

        # Filter out everything that's not a string
        local JDATA=$(jq -r 'to_entries | map(select(.value | type=="string")) | from_entries' $1)

        # Convert to key value and surround value with quotes if it has a space in it
        local JDATA=$(jq -r 'to_entries[] | "\(.key)=\(if (.value | contains(" ")) then "\""+.value+"\"" else .value end)"' /dev/stdin <<<"${JDATA}")

        # Source the contents of json
        source /dev/stdin <<<"${JDATA}"

    } || {

        return 1

    }

}


#######################################
# load_config_file sources json config file
# Arguments:
#   First positional argument is the json file path
#######################################
function load_config_file() {

    json_to_bash_variables $1
    log_on_failure "$?" "Failed to load config $1"

}


#######################################
# get_json_value return the value of a key in a json file
# Arguments:
#   First positional argument is the json file path
#   Second positional argument is the key in json object
# Returns:
#   Value for the input key in the json file
#######################################
function get_json_value() {

    if [[ $# -lt 2 ]]; then
        log_error "function get_json_value needs two arguments"
        exit 1
    fi

    jq -rc ".$2?" "$1"

}

#######################################
# set_json_value return the value of a key in a json file
# Arguments:
#   First positional argument is the json file path
#   Second positional argument is the key in json object
#   Third positional argument is the value to be set
# Returns:
#   updated json object
#######################################
function set_json_value() {

    if [[ $# -lt 3 ]]; then
        log_error "function set_json_value needs three arguments"
        exit 1
    fi
    if [[ $3 = *"/"* ]]; then
        jq ".$2 = \"$3\"" "$1"
    else
        jq ".$2 = $3" "$1"
    fi
}


#######################################
# timed times any function/command that is passed
# Arguments:
#   A command or a function that needs to be timed
#######################################
function timed() {

    START=$(date +%s.%N)
    $@
    END=$(date +%s.%N)
    DIFF=$(echo "$END - $START" | bc)
    cecho -c 'green' "Execution Time: ${DIFF} seconds"

}


#######################################
# untar the file
# Arguments:
#   First positional argument is the input tar file path
#   Second positional argument is the output untar file path
#######################################
function untar_file()
{
    local input_file=$1
    local out_dir=$2
    create_directory -d $out_dir
    tar -zxf $input_file -C $out_dir
}

#####################################
# normalize the locale string
# Arguments:
#   First positional argument is the locale string
#####################################
locale_normalize()
{
    local _LOCALE=$1
    local _LOCALE_STR=$(sed "s/-//;s/_//" <<< "${_LOCALE}")
    local _LOCALE_NORM=$(sed -e "s/\(.*\)/\L\1/" <<< "${_LOCALE_STR}")
    echo "${_LOCALE_NORM}"
}

