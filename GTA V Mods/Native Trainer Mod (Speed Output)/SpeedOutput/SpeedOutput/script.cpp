/*
	THIS FILE IS A PART OF GTA V SCRIPT HOOK SDK
				http://dev-c.com			
			(C) Alexander Blade 2015
*/

/*
	F4					activate
	NUM2/8/4/6			navigate thru the menus and lists (numlock must be on)
	NUM5 				select
	NUM0/BACKSPACE/F4 	back
	NUM9/3 				use vehicle boost when active
	NUM+ 				use vehicle rockets when active
*/

#include "script.h"
#include "keyboard.h"

#include <string>
#include <ctime>
#include <iostream>
#include <fstream>

std::string statusText;
DWORD statusTextDrawTicksMax;
bool statusTextGxtEntry;

void update_status_text()
{
	if (GetTickCount() < statusTextDrawTicksMax)
	{
		UI::SET_TEXT_FONT(0);
		UI::SET_TEXT_SCALE(0.55, 0.55);
		UI::SET_TEXT_COLOUR(255, 255, 255, 255);
		UI::SET_TEXT_WRAP(0.0, 1.0);
		UI::SET_TEXT_CENTRE(1);
		UI::SET_TEXT_DROPSHADOW(0, 0, 0, 0, 0);
		UI::SET_TEXT_EDGE(1, 0, 0, 0, 205);
		if (statusTextGxtEntry)
		{
			UI::_SET_TEXT_ENTRY((char *)statusText.c_str());
		} else
		{
			UI::_SET_TEXT_ENTRY("STRING");
			UI::_ADD_TEXT_COMPONENT_STRING((char *)statusText.c_str());
		}
		UI::_DRAW_TEXT(0.5, 0.5);
	}
}

void set_status_text(std::string str, DWORD time = 2500, bool isGxtEntry = false)
{
	statusText = str;
	statusTextDrawTicksMax = GetTickCount() + time;
	statusTextGxtEntry = isGxtEntry;
}

const char* pathFileName = "SpeedOutputPath.txt";
std::string path;

void main()
{
	std::ifstream pathFile(pathFileName, std::iostream::in);
	std::getline(pathFile, path);
	pathFile.close();

	while (true)
	{

		Player player = PLAYER::PLAYER_ID();
		Ped playerPed = PLAYER::PLAYER_PED_ID();

		if (PED::IS_PED_IN_ANY_VEHICLE(playerPed, TRUE))
		{
			std::ofstream file(path, std::iostream::out | std::iostream::trunc);

			float speed = ENTITY::GET_ENTITY_SPEED(PED::GET_VEHICLE_PED_IS_USING(playerPed));
			//set_status_text(std::to_string(speed));
			file << std::to_string(speed);

			file.close();
		}
	
		WAIT(0);
	}
}

void ScriptMain()
{
	srand(GetTickCount());
	main();
}
