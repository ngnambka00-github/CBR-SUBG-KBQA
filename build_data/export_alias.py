import pandas as pd

from file_utils import read_csv, write_dict_to_txt
from constants import NONE_VALUE

"""
Alias: 
    1. shortname_fullname
    2. role_department_fullname
    3. shortname_email
    4. fullname_email
    5. bod_fullname
    6. shortname_gender
    7. fullname_gender
    8. shortname_birthday
    9. fullname_birthday
    10. shortname_startworkdate
    11. fullname_startworkdate
    12. shortname_general_role
    13. fullname_general_role
    14. shortname_department_role
    15. fullname_department_role
    16. shortname_worklocation
    17. fullname_worklocation
    18. shortname_office
    19. fullname_office
    20. ftech_address
    21. office_address
    22. shortname_department
    23. fullname_department
    24. ftech_fullname
    25. ftech_website
    26. ftech_phone
"""



def export_person_alias():
    person_path = "./data/raw_kg/Employee__Person.csv"
    department_path = "./data/raw_kg/Department.csv"
    office_path = "./data/raw_kg/Office.csv"
    work_location_path = "./data/raw_kg/Work_Location.csv"

    person_organization_path = "./data/raw_kg/Employee__WORK__Organization.csv"
    person_department_path = "./data/raw_kg/Employee__WORK__Department.csv"
    person_office_path = "./data/raw_kg/Employee__WORK__Office.csv"
    person_work_location_path = "./data/raw_kg/Employee__WORK__Work_Location.csv"

    person_df = read_csv(person_path).rename(columns={"id": "employee_id"})
    department_df = read_csv(department_path).rename(columns={"id": "department_id"})
    office_df = read_csv(office_path).rename(columns={"id": "office_id"})
    location_df = read_csv(work_location_path).rename(columns={"id": "location_id"})

    person_organization_join = read_csv(person_organization_path).rename(
        columns={"start_id": "employee_id", "end_id": "oganization_id"})
    person_organization_join = pd.merge(person_organization_join, person_df[["employee_id", "name", "email"]],
                                        on="employee_id", how="left")

    person_department_join = read_csv(person_department_path).rename(
        columns={"start_id": "employee_id", "end_id": "department_id"})
    person_department_join = pd.merge(person_department_join, person_df[["employee_id", "name", "email"]],
                                      on="employee_id", how="left")
    person_department_join = pd.merge(person_department_join, department_df[["department_id", "name"]],
                                      on="department_id", how="left", suffixes=("_person", "_department"))

    person_office_join = read_csv(person_office_path).rename(
        columns={"start_id": "employee_id", "end_id": "office_id"})
    person_office_join = pd.merge(person_office_join, person_df[["employee_id", "name", "email"]],
                                  on="employee_id", how="left")
    person_office_join = pd.merge(person_office_join, office_df[["office_id", "name"]],
                                  on="office_id", how="left", suffixes=("_person", "_office"))

    person_location_join = read_csv(person_work_location_path).rename(
        columns={"start_id": "employee_id", "end_id": "location_id"})
    person_location_join = pd.merge(person_location_join, person_df[["employee_id", "name", "email"]],
                                    on="employee_id", how="left")
    person_location_join = pd.merge(person_location_join, location_df[["location_id", "name"]],
                                    on="location_id", how="left", suffixes=("_person", "_location"))
    result_alias = {
        "shortname_fullname": [], "role_department_fullname": [], "bod_fullname": [],
        "shortname_email": [], "fullname_email": [],
        "shortname_gender": [], "fullname_gender": [], "shortname_birthday": [], "fullname_birthday": [],
        "shortname_startworkdate": [], "fullname_startworkdate": [],
        "shortname_general_role": [], "fullname_general_role": [],
        "shortname_department_role": [], "fullname_department_role": [],
        "shortname_worklocation": [], "fullname_worklocation": [],
        "shortname_office": [], "fullname_office": [],
        "shortname_department": [], "fullname_department": [],
    }

    for idx in range(person_df.shape[0]):
        fullname = person_df.at[idx, "name"].strip()
        email = person_df.at[idx, "email"].strip()
        shortname = email.split("@")[0]
        gender = person_df.at[idx, "gender"].strip()
        birthday = person_df.at[idx, "birthday"].strip().split("T")[0]
        start_work_date = person_df.at[idx, "workStartDate"].strip().split("T")[0]

        result_alias["shortname_fullname"].append(f"{shortname}|{fullname}")
        result_alias["shortname_email"].append(f"{shortname}|{email}")
        result_alias["fullname_email"].append(f"{fullname}|{email}")
        result_alias["shortname_gender"].append(f"{shortname}|{gender}")
        result_alias["fullname_gender"].append(f"{fullname}|{gender}")
        result_alias["shortname_birthday"].append(f"{shortname}|{birthday}")
        result_alias["fullname_birthday"].append(f"{fullname}|{birthday}")
        result_alias["shortname_startworkdate"].append(f"{shortname}|{start_work_date}")
        result_alias["fullname_startworkdate"].append(f"{fullname}|{start_work_date}")

    # 1. shortname_general_role & 2. fullname_general_role & 3. bod_fullname
    special_employees = ["CPO - Giám đốc Sản phẩm", "CBO - Giám đốc Kinh doanh", "CIO - Giám đốc Thông tin",
                         "CCO - Giám đốc Sáng tạo", "CTO - Giám đốc Công nghệ", "COO - Giám đốc Thường trực",
                         "CIA - Giám đốc Nội chính", "CEO - Giám đốc điều hành"]
    for idx in range(person_organization_join.shape[0]):
        fullname = person_organization_join.at[idx, "name"].strip()
        shortname = person_organization_join.at[idx, "email"].strip().split("@")[0]
        role = person_organization_join.at[idx, "role"].strip()

        if role in special_employees:
            roles = [r.strip() for r in role.split("-")]
            result_alias["bod_fullname"].append(f"{roles[0]}|{fullname}")
            result_alias["bod_fullname"].append(f"{roles[1]}|{fullname}")
            result_alias["bod_fullname"].append(f"{role}|{fullname}")

        result_alias["shortname_general_role"].append(f"{shortname}|{role}")
        result_alias["fullname_general_role"].append(f"{fullname}|{role}")

    # 4. shortname_department_role, 5. fullname_department_role, 6. shortname_department 7. fullname_department,
    # 8. role_department_fullname
    ignore_role_department = ["Nhân viên", "CPO - Giám đốc Sản phẩm", "CBO - Giám đốc Kinh doanh",
                              "CIO - Giám đốc Thông tin", "CCO - Giám đốc Sáng tạo", "CTO - Giám đốc Công nghệ",
                              "COO - Giám đốc Thường trực", "CIA - Giám đốc Nội chính", "CEO - Giám đốc điều hành"]
    for idx in range(person_department_join.shape[0]):
        fullname = person_department_join.at[idx, "name_person"].strip()
        department_name = person_department_join.at[idx, "name_department"].strip()
        shortname = person_department_join.at[idx, "email"].strip().split("@")[0]
        role = person_department_join.at[idx, "role"].strip()

        result_alias["shortname_department_role"].append(f"{shortname}|{department_name}|{role}")
        result_alias["fullname_department_role"].append(f"{fullname}|{department_name}|{role}")
        result_alias["shortname_department"].append(f"{fullname}|{department_name}")
        result_alias["fullname_department"].append(f"{shortname}|{department_name}")

        if role not in ignore_role_department:
            result_alias["role_department_fullname"].append(f"{role}|{department_name}|{fullname}")

    # 9. shortname_office 10. fullname_office
    for idx in range(person_office_join.shape[0]):
        fullname = person_office_join.at[idx, "name_person"].strip()
        office_name = person_office_join.at[idx, "name_office"].strip()
        shortname = person_office_join.at[idx, "email"].strip().split("@")[0]

        result_alias["shortname_office"].append(f"{shortname}|{office_name}")
        result_alias["fullname_office"].append(f"{fullname}|{office_name}")

    # 11. shortname_worklocation, fullname_worklocation
    for idx in range(person_location_join.shape[0]):
        fullname = person_location_join.at[idx, "name_person"].strip()
        location_name = person_location_join.at[idx, "name_location"].strip()
        shortname = person_location_join.at[idx, "email"].strip().split("@")[0]

        result_alias["shortname_worklocation"].append(f"{shortname}|{location_name}")
        result_alias["fullname_worklocation"].append(f"{fullname}|{location_name}")

    out_path = "./data/alias/person_infor.chatette"
    write_dict_to_txt(data=result_alias, out_path=out_path)


"""
    1. ftech_address
    2. ftech_fullname
    3. ftech_website
    4. ftech_phone
    5. office_address
"""


def export_ftech_alias():
    organization_path = "./data/raw_kg/Organization.csv"
    office_path = "./data/raw_kg/Office.csv"

    organization_df = read_csv(organization_path)
    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]
    office_df = read_csv(office_path)

    result_alias = {
        "ftech_address": [],
        "ftech_fullname": [],
        "ftech_website": [],
        "ftech_phone": [],
        "office_address": []
    }
    for idx in range(organization_df.shape[0]):
        address = organization_df.at[idx, "address"].strip()
        fullname = organization_df.at[idx, "name"].strip()
        website = organization_df.at[idx, "website"].strip()
        phone = str(organization_df.at[idx, "phone"]).strip()

        for organ in organizations:
            result_alias["ftech_address"].append(f"{organ}|{address}")
            result_alias["ftech_fullname"].append(f"{organ}|{fullname}")
            result_alias["ftech_website"].append(f"{organ}|{website}")
            result_alias["ftech_phone"].append(f"{organ}|{phone}")

    # office_address
    for idx in range(office_df.shape[0]):
        office = office_df.at[idx, "name"].strip()
        address = office_df.at[idx, "address"].strip()
        if office != NONE_VALUE:
            result_alias["office_address"].append(f"{office}|{address}")

    out_path = "./data/alias/ftech_infor.chatette"
    write_dict_to_txt(data=result_alias, out_path=out_path)


if __name__ == "__main__":
    export_person_alias()
    export_ftech_alias()
