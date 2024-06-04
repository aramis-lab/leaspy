////////////////////////////////////////
// THE MODEL
////////////////////////////////////////

// constant
var AGE_PTS = 500; // number of point in ages grid
var AGE_MIN = 60; // TODO? use custom values depending tau_mean & tau_std ?
var AGE_MAX = 110;

var BIRTH_DATE_DEFAULT = '1950-01-01';
var BIRTH_DATE_OLDEST = '1920-01-01';
var BIRTH_DATE_YOUNGEST = '2000-01-01';

var N_TAU_STD = 3; // ~99% CI
var N_XI_STD = 2; // ~95% CI
var N_SOURCES_STD = 2; // ~95% CI
var N_TAU_STEPS = 100;
var N_XI_STEPS = 100;
var N_SOURCES_STEPS = 100;

var DECIMALS_TAU = 1;
var DECIMALS_XI = 2;
var DECIMALS_SOURCES = 2;
var DECIMALS_Y = 2; // in tooltip
var DECIMALS_AGE = 1; // in tooltip

var ORDINAL_EXPECTATION = true; // otherwise simply plot the ordinal MLE
var ORDINAL_NORMALIZE = false; // should we scale all ordinal features into [0, 1] (REDO plot or personalization if true)

// seaborn "muted" color palette (10 colors)
var PLOT_COLORS = [
  '#4878d0',
  '#ee854a',
  '#6acc64',
  '#d65f5f',
  '#956cb4',
  '#8c613c',
  '#dc7ec0',
  '#797979',
  '#d5bb67',
  '#82c6e2',
]

// global variables
var model = {};
var ages = null;
var hot = null; // global var for Handsontable object
var myChart = null; // global Chart.js object

// event listeners
document.getElementById("file_model").onchange = function() {
  var files = document.getElementById('file_model').files;

  if (files.length <= 0) {
    return false;
  }

  var fr = new FileReader();
  fr.onload = initModel;
  fr.readAsText(files.item(0));
};

document.getElementById("file_patients").onchange = function() {
  var files = document.getElementById('file_patients').files;

  if (files.length <= 0) {
    return false;
  }

  var fr = new FileReader();
  fr.onload = loadExistingPatient;
  fr.readAsText(files.item(0));
};

let triggerInput = (id, value, min, max, step) => {
  var input = document.createElement('input');
  // value is always "precise" (do not round it)
  min = value - step * Math.floor((value-min)/step);
  max = value + step * Math.floor((max-value)/step);

  input.setAttribute('type', 'range');
  input.setAttribute('id', id);
  input.setAttribute('min', min);
  input.setAttribute('max', max);
  input.setAttribute('step', step);
  input.setAttribute('oninput', 'onTriggerChange()');
  input.setAttribute('value', value);

  return input
}

let triggerCol = (title, id, value, min, max, step) => {
  var title_p = document.createElement('p');
  //title_p.innerText = title + ': ' + value;
  title_p.innerText = '';

  var input = triggerInput(id, value, min, max, step);

  var col = document.createElement('div');
  col.setAttribute('class', 'col-md-12');
  col.appendChild(title_p);
  col.appendChild(input);

  return col
}

let initTriggers = (model_p) => {
  var param = model_p['parameters'];
  var triggersCol = document.getElementById('triggers');

  // Temporal shift
  var min = -N_TAU_STD * param['tau_std'];
  var max =  N_TAU_STD * param['tau_std'];
  var step = Number(((max - min) / N_TAU_STEPS).toFixed(DECIMALS_TAU));
  var tempCol = triggerCol('Time shift', 'time_shift', 0, min, max, step);
  triggersCol.appendChild(tempCol);

  // Acceleration factor
  var min = -N_XI_STD * param['xi_std'];
  var max =  N_XI_STD * param['xi_std'];
  var step = Number(((max - min) / N_XI_STEPS).toFixed(DECIMALS_XI));
  var accCol = triggerCol('Acceleration factor', 'acc_factor', 0, min, max, step);
  triggersCol.appendChild(accCol);

  // Space shifts
  for(var i=0; i<model_p['source_dimension']; ++i) {
    var min = -N_SOURCES_STD * param['sources_std'];
    var max =  N_SOURCES_STD * param['sources_std'];
    var step = Number(((max - min) / N_SOURCES_STEPS).toFixed(DECIMALS_SOURCES));
    var spaceCol = triggerCol('Geometric pattern '+ (i+1), 'geom_'+i, 0, min, max, step);
    triggersCol.appendChild(spaceCol);
  }
}

let clearPage = () => {
  var canvasDiv = document.getElementById("canvas");
  while (canvasDiv.firstChild) {
    canvasDiv.removeChild(canvasDiv.firstChild);
  }

  var canvas = document.createElement('canvas');
  canvas.setAttribute('id', 'myChart');
  canvasDiv.appendChild(canvas);

  var triggers = document.getElementById("triggers");
  while (triggers.firstChild) {
    triggers.removeChild(triggers.firstChild);
  }
}

let initPlot = () => {
  var indivParameters = getTriggerValues();
  ages = []
  var age_step = (AGE_MAX - AGE_MIN)/AGE_PTS;
  for(var a = AGE_MIN; a <= AGE_MAX; a += age_step) {
    ages.push(a);
  }

  var data = compute_values(ages, model, indivParameters);
  var datasets = [];
  var realFeatureLabels = model["features"];
  var dimension = model["dimension"];
  var emptyFeaturesLabels = new Array(dimension);
  for(var i=0; i<dimension; ++i) {
    realFeatureLabels[i] = realFeatureLabels[i][0].toUpperCase() + realFeatureLabels[i].slice(1);
    emptyFeaturesLabels[i] = "";
  }

  for(var j=0; j<2; ++j) {
    var borderWidth = ( j==0 ? 3: 1.5);
    var borderDash = ( j==0 ? undefined : [8,5]);
    var featureNames = ( j==0 ? realFeatureLabels: emptyFeaturesLabels);

    for(var i=0; i<data.length; ++i) {
      var dataset = {
        label: featureNames[i],
        data: convertData(ages, data[i]),
        fill: 'rgba(0, 0, 0, 0)',
        showLine: true,
        borderDash: borderDash,
        borderWidth: borderWidth,
        borderColor: PLOT_COLORS[i % PLOT_COLORS.length],
        pointRadius: 0
      }

      datasets.push(dataset);
    }
  }

  var ctx = document.getElementById("myChart");

  myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: datasets
    },
    options: {
      maintainAspectRatio: false,
      tooltips: {
        mode: 'index', // 'index'
        intersect: true,
        filter: function(tooltipItem, data) {
          // only keep plain curves items
          return data.datasets[tooltipItem.datasetIndex].label != '';
        },
        callbacks: {
          title: function(tooltipItems, data) {
            var r_age = tooltipItems[0].xLabel.toFixed(DECIMALS_AGE);
            return 'Age: ' + r_age
          },
          label: function(tooltipItem, data) {
            // Prefix feature name + round values
            var label = data.datasets[tooltipItem.datasetIndex].label || '';
            var data_ix = tooltipItem.index;
            if(label) {
              label += ': ';
            }
            var val = tooltipItem.yLabel.toFixed(DECIMALS_Y);
            label += val

            // We also fetch the value of dashed curve (same age index, feature is shifted by nb of features)
            var other_val = data.datasets[tooltipItem.datasetIndex + dimension].data[data_ix].y.toFixed(DECIMALS_Y);
            if(other_val != val) {
              label += ' (' + other_val + ')'
            }

            return label;
          }
        }
      },
      legend: {
        labels: {
          filter: function(label) {
            return label.text != '';
          },
          fontSize: 20
        },
      },
      hover: {
        mode: 'index',
        axis: 'x',
        intersect: true
      },
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero:true
          }
        }],
        xAxes: [{
          ticks: {
            min: AGE_MIN,
            max: AGE_MAX
          }
        }],
      },
      animation: {
        duration: 0
      }
    }
  });
}

let replaceInfinityString = (key, value) => {
  if(value == "Infinity"){
    return Infinity;
  }
  return value;
}

let initModel = (e) => {

  var model_raw_json = e.target.result;

  // launch a background task to fetch model derived parameters from python Leaspy library
  // (it would be too cumbersome to manually compute those here js)
  $.ajax({
    async: true,
    type: 'POST',
    contentType: 'application/json',
    data: model_raw_json, // we send this before any post-treatment (esp. regarding Infinity)
    dataType: 'json',
    url: '/model/load',
    success: updateModelDerivedParameters
  });

  // support 'Infinity' in JSON (for deltas in ordinal models)
  var model_json = model_raw_json.replaceAll('Infinity', '"Infinity"')
  model = JSON.parse(model_json, replaceInfinityString);
  // Tweaks so that univariate model is also supported
  model['dimension'] = model['features'].length;
  if(!('source_dimension' in model)){
    model['source_dimension'] = 0;
  }
  reinitModel(model);
}

let updateModelDerivedParameters = (model_derived_params) => {
  // handle errors
  if('error' in model_derived_params){
    alert(model_derived_params['error'])
  }
  // global update of model parameters
  for(var p in model_derived_params){
    model['parameters'][p] = model_derived_params[p]
  }
}

let reinitModel = (params) => {
  clearPage();
  initTriggers(params);
  initPlot();
  onTriggerChange();
}

////////////////////////////////////////
// INITIALIZATION OF NEW PATIENT
////////////////////////////////////////

let resetPatientButton = (individualData) => {
  var patient = document.getElementById("patient");
  while (patient.firstChild) {
    patient.removeChild(patient.firstChild);
  }

  patient.innerText = 'Birthday';
  var input = document.createElement('input');
  input.setAttribute('type', 'date');
  input.setAttribute('id', 'start');
  input.setAttribute('name', 'trip-start');
  if(individualData === undefined) {
    input.setAttribute('value', BIRTH_DATE_DEFAULT);
  } else {
    input.setAttribute('value', individualData['birthday']);
  }

  input.setAttribute('min', BIRTH_DATE_OLDEST);
  input.setAttribute('max', BIRTH_DATE_YOUNGEST);
  patient.appendChild(input);

  var addRow = document.createElement('button');
  addRow.setAttribute('type', 'button');
  addRow.setAttribute('class','btn btn-info btn-sm');
  addRow.setAttribute('onclick', 'addRow()');
  addRow.style.margin = '10px';
  addRow.innerText = 'Add a line';
  patient.appendChild(addRow);

  var removeRow = document.createElement('button');
  removeRow.setAttribute('type', 'button');
  removeRow.setAttribute('class', 'btn btn-warning btn-sm');
  removeRow.setAttribute('onclick', 'removeRow()');
  removeRow.style.margin = '10px';
  removeRow.innerText = 'Delete last line';
  patient.appendChild(removeRow);

  var personalize = document.createElement('button');
  personalize.setAttribute('type', 'button');
  personalize.setAttribute('class', 'btn btn-success btn-sm');
  personalize.setAttribute('onclick', 'personalize()');
  personalize.style.margin = '10px';
  personalize.innerText = 'Personalize';
  patient.appendChild(personalize);

  var reset = document.createElement('button');
  reset.setAttribute('type', 'button');
  reset.setAttribute('class', 'btn btn-danger btn-sm');
  reset.setAttribute('onclick', 'resetTriggerValues(); reinitModel(model)'); // reset plot without individual data
  reset.style.margin = '10px';
  reset.innerText = 'Reinitialize';
  patient.appendChild(reset);
}

let initTable = (model, individualData) => {
  var hotElement = document.querySelector('#table');

  var columns = [{
    data: 'asOf',
    type: 'date',
    dateFormat: 'DD/MM/YYYY',
    correctFormat: false,
  }];
  var headers = ["Date"];

  for(var i=0; i<model['dimension']; ++i){
    headers.push(model["features"][i]);
    columns.push({
      data: 'val'+i,
      type: 'numeric',
      numericFormat: {
        pattern: '0.00'
      }
    });
  }

  if (individualData === undefined) {
    var today = new Date();
    var day_padded = ('0' + today.getDate()).slice(-2)
    var month_padded = ('0' + (today.getMonth() + 1)).slice(-2)
    var date = day_padded + '/' + month_padded + '/' + today.getFullYear();
    var dataObject = {asOf: date};

    for(var i=0; i<model['dimension']; ++i){
      dataObject["val"+i] = '';
    }
    dataObject = [dataObject]

  } else {
    var dataObject = [];
    for(var i=0; i<individualData['visits'].length;++i) {
      var unitDataObject = {asOf: individualData['visits'][i][0]};
      for(var j=0; j<model['dimension']; ++j){
        unitDataObject["val"+j] = individualData['visits'][i][j+1];
      }

      dataObject.push(unitDataObject);

    }
  }

  var hotSettings = {
    licenseKey: 'non-commercial-and-evaluation',
    data: dataObject,
    columns: columns,
    stretchH: 'all',
    width: 805,
    autoWrapRow: true,
    maxRows: 22,
    rowHeaders: true,
    colHeaders: headers,
    fillHandle: {
    direction: 'vertical',
    autoInsertRow: true
    }
  };

  hot = new Handsontable(hotElement, hotSettings);
}

let initIndividualData = () => {
  resetPatientButton();
  initTable(model);
}

let loadExistingPatient = (e) => {
  var individualData = JSON.parse(e.target.result);
  resetPatientButton(individualData);
  initTable(model, individualData);
}
