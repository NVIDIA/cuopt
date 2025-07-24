/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*=====================================================================================

    Filename:     ReaderMPS.cpp

    Description:
        Version:  1.0

    Author:       Peng Lin, penglincs@outlook.com

    Organization: Shaowei Cai Group,
                  State Key Laboratory of Computer Science,
                  Institute of Software, Chinese Academy of Sciences,
                  Beijing, China

=====================================================================================*/

#include "ReaderMPS.h"

ReaderMPS::ReaderMPS(ModelConUtil* _modelConUtil, ModelVarUtil* _modelVarUtil)
  : modelConUtil(_modelConUtil), modelVarUtil(_modelVarUtil), integralityMarker(false)
{
}

ReaderMPS::~ReaderMPS() {}

void ReaderMPS::Read(const char* _filename)
{
  ifstream infile(_filename);
  string modelName;
  string tempStr;
  char conType;
  string conName;
  string inverseConName;
  size_t inverseConIdx;
  size_t conIdx;
  string varName;
  Value coefficient;
  Value rhs;
  string varType;
  Value inputBound;
  if (!infile) {
    printf("o The input filename %s is invalid.\n", _filename);
    exit(-1);
  }
  while (getline(infile, readLine))  // NAME section
  {
    if (readLine[0] == '*' || readLine.length() < 1) continue;
    if (readLine[0] == 'R' || readLine[0] == 'O') break;
    IssSetup();
    if (!(iss >> tempStr >> modelName)) continue;
    if (tempStr != "NAME") PrintfError(readLine);
    printf("c Model name: %s\n", modelName.c_str());
  }
  if (readLine[0] == 'O') {
    if (readLine.find("MAX") != string::npos) modelConUtil->MIN = -1;
    while (getline(infile, readLine)) {
      if (readLine[0] == '*' || readLine.length() < 1) continue;
      if (readLine[0] == 'R') break;
      IssSetup();
      iss >> tempStr;
      cout << tempStr << endl;
      if (tempStr == "MAX") modelConUtil->MIN = -1;
    }
  }
  modelConUtil->conSet.emplace_back("", 0);  // obj
  while (getline(infile, readLine))          // ROWS section
  {
    if (readLine[0] == '*' || readLine.length() < 1) continue;
    if (readLine[0] == 'C') break;
    IssSetup();
    if (!(iss >> conType >> conName))
      if (!IsBlank(readLine))
        PrintfError(readLine);
      else
        continue;
    if (conType == 'L')
      conIdx = modelConUtil->MakeCon(conName);
    else if (conType == 'E') {
      conIdx                                      = modelConUtil->MakeCon(conName);
      modelConUtil->conSet[conIdx].isEqual        = true;
      inverseConName                              = conName + "!";
      inverseConIdx                               = modelConUtil->MakeCon(inverseConName);
      modelConUtil->conSet[inverseConIdx].isEqual = true;
    } else if (conType == 'G') {
      conIdx                               = modelConUtil->MakeCon(conName);
      modelConUtil->conSet[conIdx].isLarge = true;
    } else {
      assert(conType == 'N');  // type=='N',this con is obj
      if (modelConUtil->objName != "") PrintfError(readLine);
      modelConUtil->objName = conName;
    }
  }
  while (getline(infile, readLine))  // COLUMNS section
  {
    if (readLine[0] == '*' || readLine.length() < 1) continue;
    if (readLine[0] == 'R') break;
    IssSetup();
    if (!(iss >> varName >> conName))
      if (!IsBlank(readLine))
        PrintfError(readLine);
      else
        continue;
    if (conName == "\'MARKER\'") {
      iss >> tempStr;
      if (tempStr != "\'INTORG\'" && tempStr != "\'INTEND\'") PrintfError(readLine);
      integralityMarker = !integralityMarker;
      continue;
    }
    iss >> coefficient;
    conIdx = modelConUtil->GetConIdx(conName);
    PushCoeffVarIdx(conIdx, coefficient, varName);
    if (modelConUtil->conSet[conIdx].isEqual) PushCoeffVarIdx(conIdx + 1, -coefficient, varName);
    if (iss >> conName) {
      iss >> coefficient;
      conIdx = modelConUtil->GetConIdx(conName);
      PushCoeffVarIdx(conIdx, coefficient, varName);
      if (modelConUtil->conSet[conIdx].isEqual) PushCoeffVarIdx(conIdx + 1, -coefficient, varName);
    }
  }
  while (getline(infile, readLine))  // RHS  section
  {
    if (readLine[0] == '*' || readLine.length() < 1) continue;
    if (readLine[0] == 'B' || readLine[0] == 'E') break;
    if (readLine[0] == 'R' || readLine[0] == 'S')  // do not handle RANGS and SOS
      PrintfError(readLine);
    IssSetup();
    if (!(iss >> tempStr >> conName >> rhs))
      if (!IsBlank(readLine))
        PrintfError(readLine);
      else
        continue;
    if (conName.length() < 1) continue;
    conIdx                           = modelConUtil->GetConIdx(conName);
    modelConUtil->conSet[conIdx].RHS = rhs;
    if (modelConUtil->conSet[conIdx].isEqual) modelConUtil->conSet[conIdx + 1].RHS = -rhs;

    if (iss >> conName) {
      iss >> rhs;
      conIdx                           = modelConUtil->GetConIdx(conName);
      modelConUtil->conSet[conIdx].RHS = rhs;
      if (modelConUtil->conSet[conIdx].isEqual) modelConUtil->conSet[conIdx + 1].RHS = -rhs;
    }
  }
  while (getline(infile, readLine))  // BOUNDS section
  {
    if (readLine[0] == '*' || readLine.length() < 1) continue;
    if (readLine[0] == 'E') break;
    if (readLine[0] == 'I')  // do not handle INDICATORS
      PrintfError(readLine);
    IssSetup();
    if (!(iss >> varType >> tempStr >> varName))
      if (!IsBlank(readLine))
        PrintfError(readLine);
      else
        continue;
    iss >> inputBound;
    if (modelVarUtil->name2idx.find(varName) != modelVarUtil->name2idx.end()) {
      auto& var = modelVarUtil->GetVar(varName);
      if (var.type == VarType::Binary) {
        var.SetType(VarType::Integer);
        var.SetUpperBound(InfiniteUpperBound);
      }
      if (varType == "UP")
        var.SetUpperBound(inputBound);
      else if (varType == "LO")
        var.SetLowerBound(inputBound);
      else if (varType == "BV") {
        var.SetType(VarType::Binary);
        var.SetUpperBound(1.0);
        var.SetLowerBound(0.0);
      } else if (varType == "LI")
        var.SetLowerBound(inputBound);
      else if (varType == "UI")
        var.SetUpperBound(inputBound);
      else if (varType == "FX") {
        var.SetLowerBound(inputBound);
        var.SetUpperBound(inputBound);
        var.SetType(VarType::Fixed);
      } else if (varType == "FR") {
        var.SetUpperBound(InfiniteUpperBound);
        var.SetLowerBound(InfiniteLowerBound);
      } else if (varType == "MI")
        var.SetLowerBound(InfiniteLowerBound);
      else if (varType == "PL")
        var.SetUpperBound(InfiniteUpperBound);
    } else
      continue;
  }
  infile.close();
  for (conIdx = 1; conIdx < modelConUtil->conSet.size(); ++conIdx) {
    auto& con = modelConUtil->conSet[conIdx];
    if (con.isLarge) {
      for (Value& inverseCoefficient : con.coeffSet)
        inverseCoefficient = -inverseCoefficient;
      con.RHS = -con.RHS;
    }
  }
  modelVarUtil->objBias = -modelConUtil->conSet[0].RHS;
  modelConUtil->conNum  = modelConUtil->conSet.size();
  modelVarUtil->varNum  = modelVarUtil->varSet.size();

  if (!TightenBound() || !TightBoundGlobally()) {
    printf("c model is infeasible.\n");
    exit(-1);
  }

  SetVarType();
  SetVarIdx2ObjIdx();
}

inline void ReaderMPS::IssSetup()
{
  iss.clear();
  iss.str(readLine);
  iss.seekg(0, ios::beg);
}

void ReaderMPS::PushCoeffVarIdx(const size_t _conIdx, Value _coeff, const string& _varName)
{
  auto& con      = modelConUtil->conSet[_conIdx];
  size_t _varIdx = modelVarUtil->MakeVar(_varName, integralityMarker);
  auto& var      = modelVarUtil->GetVar(_varIdx);

  var.conIdxSet.push_back(_conIdx);
  var.posInCon.push_back(con.varIdxSet.size());
  if (_conIdx == 0) _coeff *= modelConUtil->MIN;
  con.coeffSet.push_back(_coeff);
  con.varIdxSet.push_back(_varIdx);
  con.posInVar.push_back(var.conIdxSet.size() - 1);
}

bool ReaderMPS::TightenBound()
{
  for (size_t conIdx = 1; conIdx < modelConUtil->conNum; ++conIdx) {
    auto& modelCon = modelConUtil->conSet[conIdx];
    if (modelCon.varIdxSet.size() == 1) TightenBoundVar(modelCon);
    if (modelCon.varIdxSet.size() == 0) {
      assert(modelCon.coeffSet.size() == 0 && modelCon.posInVar.size() == 0);
      if (modelCon.RHS + 1e-6 >= 0) {
        modelCon.inferSAT = true;
        deleteConNum++;
      } else {
        printf("c con.rhs %lf\n", modelCon.RHS);
        return false;
      }
    }
  }
  return true;
}

void ReaderMPS::TightenBoundVar(ModelCon& modelCon)
{
  Value coeff    = modelCon.coeffSet[0];
  auto& modelvar = modelVarUtil->GetVar(modelCon.varIdxSet[0]);
  Value newBound = (modelCon.RHS + FeasibilityTol) / coeff;
  if (coeff > 0 && newBound < modelvar.upperBound)  // x <= bound
    modelvar.SetUpperBound(newBound);
  else if (coeff < 0 && modelvar.lowerBound < newBound)  // x >= bound
    modelvar.SetLowerBound(newBound);
}

bool ReaderMPS::TightBoundGlobally()
{
  for (auto& modelVar : modelVarUtil->varSet)
    if (modelVar.IsFixed()) {
      modelVar.SetType(VarType::Fixed);
      fixedIdxs.push_back(modelVar.idx);
    }
  while (fixedIdxs.size() > 0) {
    size_t removeVarIdx = fixedIdxs.back();
    fixedIdxs.pop_back();
    deleteVarNum++;
    ModelVar& removeVar  = modelVarUtil->GetVar(removeVarIdx);
    Value removeVarValue = removeVar.lowerBound;
    for (size_t termIdx = 0; termIdx < removeVar.conIdxSet.size(); termIdx++) {
      size_t conIdx                = removeVar.conIdxSet[termIdx];
      size_t posInCon              = removeVar.posInCon[termIdx];
      ModelCon& modelCon           = modelConUtil->GetCon(conIdx);
      Value coeff                  = modelCon.coeffSet[posInCon];
      size_t movedVarIdx           = modelCon.varIdxSet.back();
      Value movedCoeff             = modelCon.coeffSet.back();
      size_t movedPosInVar         = modelCon.posInVar.back();
      modelCon.varIdxSet[posInCon] = movedVarIdx;
      modelCon.coeffSet[posInCon]  = movedCoeff;
      modelCon.posInVar[posInCon]  = movedPosInVar;
      ModelVar& movedVar           = modelVarUtil->GetVar(movedVarIdx);
      assert(movedVar.conIdxSet[movedPosInVar] == conIdx);
      movedVar.posInCon[movedPosInVar] = posInCon;
      modelCon.varIdxSet.pop_back();
      modelCon.coeffSet.pop_back();
      modelCon.posInVar.pop_back();
      if (conIdx == 0)
        modelVarUtil->objBias += coeff * removeVarValue;
      else {
        modelCon.RHS -= coeff * removeVarValue;
        if (modelCon.varIdxSet.size() == 1) {
          TightenBoundVar(modelCon);
          ModelVar& relatedVar = modelVarUtil->GetVar(modelCon.varIdxSet[0]);
          if (relatedVar.type != VarType::Fixed && relatedVar.IsFixed()) {
            relatedVar.SetType(VarType::Fixed);
            fixedIdxs.push_back(relatedVar.idx);
            inferVarNum++;
          }
        } else if (modelCon.varIdxSet.size() == 0) {
          assert(modelCon.coeffSet.size() == 0 && modelCon.posInVar.size() == 0);
          if (modelCon.RHS + 1e-2 >= 0) {
            modelCon.inferSAT = true;
            deleteConNum++;
          } else {
            printf("c con.rhs %lf\n", modelCon.RHS);
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool ReaderMPS::SetVarType()
{
  for (size_t varIdx = 0; varIdx < modelVarUtil->varNum; varIdx++) {
    auto& modelVar   = modelVarUtil->GetVar(varIdx);
    modelVar.termNum = modelVar.conIdxSet.size();
    if (modelVar.lowerBound >= modelVar.upperBound + FeasibilityTol) {
      printf(
        "c %s LB: %lf; UB: %lf\n", modelVar.name.c_str(), modelVar.lowerBound, modelVar.upperBound);
      exit(-1);
    }
    if (modelVar.IsFixed()) {
      modelVarUtil->fixedNum++;
      modelVar.SetType(VarType::Fixed);
    } else if (modelVar.IsBinary()) {
      modelVarUtil->binaryNum++;
      modelVar.SetType(VarType::Binary);
    } else if (modelVar.type == VarType::Integer) {
      modelVarUtil->integerNum++;
    } else {
      modelVar.SetType(VarType::Real);
      modelVarUtil->realNum++;
    }
  }
  for (size_t conIdx = 0; conIdx < modelConUtil->conNum; conIdx++) {
    auto& modelCon   = modelConUtil->GetCon(conIdx);
    modelCon.termNum = modelCon.varIdxSet.size();
    if (modelCon.inferSAT) assert(modelCon.termNum == 0);
  }
  if (modelVarUtil->integerNum > 0 || modelVarUtil->realNum > 0) modelVarUtil->isBin = false;
  return true;
}

void ReaderMPS::SetVarIdx2ObjIdx()
{
  modelVarUtil->varIdx2ObjIdx.resize(modelVarUtil->varNum, -1);
  const auto& modelObj = modelConUtil->conSet[0];
  for (size_t idx = 0; idx < modelObj.termNum; ++idx)
    modelVarUtil->varIdx2ObjIdx[modelObj.varIdxSet[idx]] = idx;
}