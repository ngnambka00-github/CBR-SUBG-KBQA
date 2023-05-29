import os
from typing import Text

import pandas as pd

from file_utils import read_csv, write_to_txt, read_from_txt
from constants import NONE_VALUE


# 1. employee_person: have_email, have_short_name, have_birthday, have_gender, work_start_date
def write_kg_employee_person() -> None:
    input_path = "./data/raw_kg/Employee__Person.csv"
    data_df = read_csv(input_path)

    tags_have_email = []
    tags_have_short_name = []
    tags_have_birthday = []
    tags_have_gender = []
    tags_work_start_date = []
    for idx in range(data_df.shape[0]):
        name = data_df.at[idx, "name"].strip()
        email = data_df.at[idx, "email"].strip()
        short_name = email.split("@")[0]
        birthday = data_df.at[idx, "birthday"].strip().split("T")[0]
        gender = data_df.at[idx, "gender"].strip()
        work_start_date = data_df.at[idx, "workStartDate"].strip().split("T")[0]

        tags_have_email.append(f"{name}|have_email|{email}")
        tags_have_short_name.append(f"{name}|have_short_name|{short_name}")
        tags_have_birthday.append(f"{name}|have_birthday|{birthday}")
        tags_have_gender.append(f"{name}|have_gender|{gender}")
        tags_work_start_date.append(f"{name}|work_start_date|{work_start_date}")

    results = []
    results.extend(tags_have_email)
    results.extend(tags_have_short_name)
    results.extend(tags_have_birthday)
    results.extend(tags_have_gender)
    results.extend(tags_work_start_date)

    out_path = "data/kg/process/employee_person.txt"
    write_to_txt(data=results, out_path=out_path)


# 2. employee_WORK_department: work_depart, have_role_in_depart
def write_kg_employee_work_department() -> None:
    join_path = "./data/raw_kg/Employee__WORK__Department.csv"
    employee_path = "./data/raw_kg/Employee__Person.csv"
    depart_path = "./data/raw_kg/Department.csv"

    join_df = read_csv(join_path).rename(columns={"start_id": "employee_id", "end_id": "depart_id"})
    employee_df = read_csv(employee_path).rename(columns={"id": "employee_id"})
    depart_df = read_csv(depart_path).rename(columns={"id": "depart_id"})

    join_df = pd.merge(join_df, employee_df[["employee_id", "name"]], on="employee_id", how="left")
    join_df = pd.merge(join_df, depart_df[["depart_id", "name"]], on="depart_id", how="left",
                       suffixes=("_person", "_depart"))

    tags_work_depart = []
    tags_have_role_in_depart = []
    special_roles = ["CPO - Giám đốc Sản phẩm", "CBO - Giám đốc Kinh doanh",
                     "CIO - Giám đốc Thông tin", "CCO - Giám đốc Sáng tạo", "CTO - Giám đốc Công nghệ",
                     "COO - Giám đốc Thường trực", "CIA - Giám đốc Nội chính", "CEO - Giám đốc điều hành"]
    for idx in range(join_df.shape[0]):
        name = join_df.at[idx, "name_person"].strip()
        depart = join_df.at[idx, "name_depart"].strip()
        role = join_df.at[idx, "role"].strip()

        tags_work_depart.append(f"{name}|work_depart|{depart}")
        tags_have_role_in_depart.append(f"{name}|have_role_in_depart|{role}")
        if role in special_roles:
            roles = [r.strip() for r in role.split("-")]
            tags_have_role_in_depart.append(f"{name}|have_role_in_depart|{roles[0]}")
            tags_have_role_in_depart.append(f"{name}|have_role_in_depart|{roles[1]}")

    results = []
    results.extend(tags_work_depart)
    results.extend(tags_have_role_in_depart)

    out_path = "data/kg/process/employee_WORK_department.txt"
    write_to_txt(data=results, out_path=out_path)


# 3. employee_WORK_organization: work_organization, have_role_in_organization
def write_kg_employee_work_organization() -> None:
    join_path = "./data/raw_kg/Employee__WORK__Organization.csv"
    employee_path = "./data/raw_kg/Employee__Person.csv"

    join_df = read_csv(join_path).rename(columns={"start_id": "employee_id"})
    employee_df = read_csv(employee_path).rename(columns={"id": "employee_id"})
    join_df = pd.merge(join_df, employee_df[["employee_id", "name"]], on="employee_id", how="left")

    tags_work_organization = []
    tags_have_role_in_organization = []
    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]
    for idx in range(join_df.shape[0]):
        name = join_df.at[idx, "name"].strip()
        role = join_df.at[idx, "role"].strip()

        tags_work_organization.append(f"{name}|work_organization|{organizations[0]}")
        tags_work_organization.append(f"{name}|work_organization|{organizations[1]}")
        tags_have_role_in_organization.append(f"{name}|have_role_in_organization|{role}")

    results = []
    results.extend(tags_work_organization)
    results.extend(tags_have_role_in_organization)

    out_path = "data/kg/process/employee_WORK_organization.txt"
    write_to_txt(data=results, out_path=out_path)


# 4. employee_WORK_office: work_office
def write_kg_employee_work_office() -> None:
    join_path = "./data/raw_kg/Employee__WORK__Office.csv"
    employee_path = "./data/raw_kg/Employee__Person.csv"
    office_path = "./data/raw_kg/Office.csv"

    join_df = read_csv(join_path).rename(columns={"start_id": "employee_id", "end_id": "office_id"})
    employee_df = read_csv(employee_path).rename(columns={"id": "employee_id"})
    office_df = read_csv(office_path).rename(columns={"id": "office_id"})

    join_df = pd.merge(join_df, employee_df[["employee_id", "name"]], on="employee_id", how="left")
    join_df = pd.merge(join_df, office_df[["office_id", "name"]], on="office_id", how="left",
                       suffixes=("_person", "_office"))

    tags_work_office = []
    for idx in range(join_df.shape[0]):
        name = join_df.at[idx, "name_person"].strip()
        office = join_df.at[idx, "name_office"].strip()
        tags_work_office.append(f"{name}|work_office|{office}")

    out_path = "data/kg/process/employee_WORK_office.txt"
    write_to_txt(data=tags_work_office, out_path=out_path)


# 5. employee_WORK_work_location: work_location
def write_kg_employee_work_location() -> None:
    join_path = "./data/raw_kg/Employee__WORK__Work_Location.csv"
    employee_path = "./data/raw_kg/Employee__Person.csv"
    location_path = "./data/raw_kg/Work_Location.csv"

    join_df = read_csv(join_path).rename(columns={"start_id": "employee_id", "end_id": "location_id"})
    employee_df = read_csv(employee_path).rename(columns={"id": "employee_id"})
    location_df = read_csv(location_path).rename(columns={"id": "location_id"})

    join_df = pd.merge(join_df, employee_df[["employee_id", "name"]], on="employee_id", how="left")
    join_df = pd.merge(join_df, location_df[["location_id", "name"]], on="location_id", how="left",
                       suffixes=("_person", "_location"))

    tags_work_location = []
    for idx in range(join_df.shape[0]):
        name = join_df.at[idx, "name_person"].strip()
        location = join_df.at[idx, "name_location"].strip()
        tags_work_location.append(f"{name}|work_location|{location}")

    out_path = "data/kg/process/employee_WORK_location.txt"
    write_to_txt(data=tags_work_location, out_path=out_path)


# 6. work_location_BELONG_TO_office: belong_to
def write_kg_work_location_belong_to_office() -> None:
    join_path = "./data/raw_kg/Work_Location__BELONG_TO__Office.csv"
    location_path = "./data/raw_kg/Work_Location.csv"
    office_path = "./data/raw_kg/Office.csv"

    join_df = read_csv(join_path).rename(columns={"start_id": "location_id", "end_id": "office_id"})
    location_df = read_csv(location_path).rename(columns={"id": "location_id"})
    office_df = read_csv(office_path).rename(columns={"id": "office_id"})

    join_df = pd.merge(join_df, location_df[["location_id", "name"]], on="location_id", how="left")
    join_df = pd.merge(join_df, office_df[["office_id", "name"]], on="office_id", how="left",
                       suffixes=("_location", "_office"))

    tags_belong_to = []
    for idx in range(join_df.shape[0]):
        location = join_df.at[idx, "name_location"].strip()
        office = join_df.at[idx, "name_office"].strip()
        if location != NONE_VALUE:
            tags_belong_to.append(f"{location}|belong_to|{office}")

    out_path = "data/kg/process/location_BELONG_TO_office.txt"
    write_to_txt(data=tags_belong_to, out_path=out_path)


# 7. office_BELONG_TO_organization: belong_to
def write_kg_office_belong_to_organization() -> None:
    join_path = "./data/raw_kg/Office__BELONG_TO__Organization.csv"
    office_path = "./data/raw_kg/Office.csv"
    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]

    join_df = read_csv(join_path).rename(columns={"start_id": "office_id"})
    office_df = read_csv(office_path).rename(columns={"id": "office_id"})
    join_df = pd.merge(join_df, office_df[["office_id", "name"]], on="office_id", how="left")

    tags_belong_to = []
    for idx in range(join_df.shape[0]):
        office = join_df.at[idx, "name"].strip()
        tags_belong_to.append(f"{office}|belong_to|{organizations[0]}")
        tags_belong_to.append(f"{office}|belong_to|{organizations[1]}")

    out_path = "data/kg/process/office_BELONG_TO_organization.txt"
    write_to_txt(data=tags_belong_to, out_path=out_path)


# 8. work_location_BELONG_TO_organization: belong_to
def write_kg_location_belong_to_organization() -> None:
    join_path = "./data/raw_kg/Work_Location__BELONG_TO__Organization.csv"
    location_path = "./data/raw_kg/Work_Location.csv"
    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]

    join_df = read_csv(join_path).rename(columns={"start_id": "location_id"})
    location_df = read_csv(location_path).rename(columns={"id": "location_id"})
    join_df = pd.merge(join_df, location_df[["location_id", "name"]], on="location_id", how="left")

    tags_belong_to = []
    for idx in range(join_df.shape[0]):
        location = join_df.at[idx, "name"].strip()
        tags_belong_to.append(f"{location}|belong_to|{organizations[0]}")
        tags_belong_to.append(f"{location}|belong_to|{organizations[1]}")

    out_path = "data/kg/process/location_BELONG_TO_organization.txt"
    write_to_txt(data=tags_belong_to, out_path=out_path)


# 9. department_BELONG_TO_organization: belong_to
def write_kg_department_belong_to_organization() -> None:
    join_path = "./data/raw_kg/Department__BELONG_TO__Organization.csv"
    depart_path = "./data/raw_kg/Department.csv"
    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]

    join_df = read_csv(join_path).rename(columns={"start_id": "depart_id"})
    depart_df = read_csv(depart_path).rename(columns={"id": "depart_id"})
    join_df = pd.merge(join_df, depart_df[["depart_id", "name"]], on="depart_id", how="left")

    tags_belong_to = []
    for idx in range(join_df.shape[0]):
        depart = join_df.at[idx, "name"].strip()
        tags_belong_to.append(f"{depart}|belong_to|{organizations[0]}")
        tags_belong_to.append(f"{depart}|belong_to|{organizations[1]}")

    out_path = "data/kg/process/depart_BELONG_TO_organization.txt"
    write_to_txt(data=tags_belong_to, out_path=out_path)


# 10. office_address_..., work_location_address_... : relation -> address
def write_kg_office_address_and_work_location_address() -> None:
    office_path = "./data/raw_kg/Office.csv"
    location_path = "./data/raw_kg/Work_Location.csv"
    office_df = read_csv(office_path)
    location_df = read_csv(location_path)

    tags_address = []
    for idx in range(office_df.shape[0]):
        office = office_df.at[idx, "name"].strip()
        if office == NONE_VALUE:
            continue
        address = office_df.at[idx, "address"].strip()
        tags_address.append(f"{office}|address|{address}")

    for idx in range(location_df.shape[0]):
        location = location_df.at[idx, "name"].strip()
        if location == NONE_VALUE:
            continue
        address = location_df.at[idx, "address"].strip()
        tags_address.append(f"{location}|address|{address}")

    # remove duplicated object
    tags_address = list(set(tags_address))

    out_path = "data/kg/process/office_and_work_location_ADDRESS_.txt"
    write_to_txt(data=tags_address, out_path=out_path)


# 11. Organization Information: has_name, has_address, has_website, has_phone
def write_kg_organization_information() -> None:
    organization_path = "./data/raw_kg/Organization.csv"
    organization_df = read_csv(organization_path)

    organizations = ["FTECH", "Công ty Công nghệ Gia Đình"]
    tags_has_name = []
    tags_has_address = []
    tags_has_website = []
    tags_has_phone = []
    for idx in range(organization_df.shape[0]):
        name = organization_df.at[idx, "name"].strip()
        address = organization_df.at[idx, "address"].strip()
        website = organization_df.at[idx, "website"].strip()
        phone = str(organization_df.at[idx, "phone"]).strip()

        for organ in organizations:
            tags_has_name.append(f"{organ}|has_name|{name}")
            tags_has_address.append(f"{organ}|has_address|{address}")
            tags_has_website.append(f"{organ}|has_website|{website}")
            tags_has_phone.append(f"{organ}|has_phone|{phone}")

    results = []
    results.extend(tags_has_name)
    results.extend(tags_has_address)
    results.extend(tags_has_website)
    results.extend(tags_has_phone)

    out_path = "data/kg/process/orgainization_HAS_INFORMATION_.txt"
    write_to_txt(data=results, out_path=out_path)


def build_full_graph(kg_folder_path: Text, out_folder_path: Text, convert_from_db: bool = False) -> None:
    if convert_from_db:
        write_kg_employee_work_department()
        write_kg_employee_work_organization()
        write_kg_employee_work_office()
        write_kg_employee_work_location()
        write_kg_work_location_belong_to_office()
        write_kg_office_belong_to_organization()
        write_kg_location_belong_to_organization()
        write_kg_department_belong_to_organization()
        write_kg_office_address_and_work_location_address()
        write_kg_organization_information()

    # merger all file path
    all_triple_kg = []
    for file_name in os.listdir(kg_folder_path):
        file_path = os.path.join(kg_folder_path, file_name)
        all_triple_kg.extend(read_from_txt(file_path))
    entities = []
    for triple in all_triple_kg:
        ere = triple.split("|")
        entities.append(ere[0])
        entities.append(ere[2])
    entities = list(set(entities))
    relations = [triple.split("|")[1] for triple in all_triple_kg]
    relations = list(set(relations))

    triple_path = os.path.join(out_folder_path, "kb.txt")
    entities_path = os.path.join(out_folder_path, "entity.txt")
    relation_path = os.path.join(out_folder_path, "relation.txt")
    write_to_txt(all_triple_kg, triple_path)
    write_to_txt(entities, entities_path)
    write_to_txt(relations, relation_path)


if __name__ == "__main__":
    build_full_graph(
        kg_folder_path="./data/kg/process",
        out_folder_path="./data/kg",
        convert_from_db=True)
