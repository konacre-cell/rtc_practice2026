using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Splines;
using Unity.Mathematics;



public class DatasetRecorderSpline : MonoBehaviour
{
    [Header(" Объект движения и путь ")]
    public Transform movingObject; // что двигаем
    public SplineContainer splinePath; // Сплайн, по которому едет робот

    [Header(" Камера и рендер")]
    public Camera captureCamera; // камера, с которой делаем снимки
    public RenderTexture renderTexture; // её RenderTexture

    [Header(" Параметры движения ")]
    public float moveSpeed = 0.5f; // скорость движения (юниты/сек)
    public bool rotateWithSpline = true; // Поворачивать ли робота по направлению движения

    [Header(" Параметры съёмки ")]
    public int captureFPS = 20; // кадров в секунду
    public int totalFrames = 300; // сколько кадров сделать (0 = бесконечно)
    public bool autoStart = true; // начать запись сразу при старте сцены
    public KeyCode startKey = KeyCode.Space; // клавиша для ручного старта
    public KeyCode stopKey = KeyCode.Escape; // клавиша для остановки

    [Header(" Сохранение ")]
    public string folderName = "Dataset_Run_Spline"; // папка внутри проекта
    public string fileNamePrefix = "frame"; // префикс имени файла

    private string savePath;
    private bool isRecording = false;
    private int frameCounter = 0;
    private float currentDistance = 0f; // Пройденная дистанция по сплайну




    void Start()
    {
        // Автопоиск компонентов
        if (movingObject == null) movingObject = transform;

        if (captureCamera == null) captureCamera = GetComponent<Camera>();
        if (captureCamera == null)
        {
            Debug.LogError(" DatasetRecorder: не назначена камера и не найдена компонента Camera на объекте");
            enabled = false;
            return;
        }

        if (renderTexture == null) renderTexture = captureCamera.targetTexture;
        if (renderTexture == null)
        {
            Debug.LogError(" DatasetRecorder: у камеры не назначен RenderTexture");
            enabled = false;
            return;
        }

        if (splinePath == null)
        {
            Debug.LogError(" DatasetRecorder: не назначен SplineContainer");
            enabled = false;
            return;
        }

        // Создание папки для сохранения 
        savePath = Path.Combine(Application.dataPath, "../", folderName);
        if (!Directory.Exists(savePath)) Directory.CreateDirectory(savePath);
        Debug.Log($" Датасет будет сохранён в: {savePath}");

        // Автостарт 
        if (autoStart) StartRecording();
    }



    void Update()
    {
        if (!isRecording)
        {
            if (Input.GetKeyDown(startKey)) StartRecording();
        }
        else
        {
            if (Input.GetKeyDown(stopKey)) StopRecording();
        }
    }



    public void StartRecording()
    {
        if (isRecording) return;
        isRecording = true;
        frameCounter = 0;
        currentDistance = 0f; // Сбрасываем дистанцию при новом старте
        Debug.Log(" Начало записи датасета по сплайну...");
        StartCoroutine(RecordingCoroutine());
    }



    public void StopRecording()
    {
        isRecording = false;
        Debug.Log($" Запись остановлена. Сохранено кадров: {frameCounter}");
    }




    IEnumerator RecordingCoroutine()
    {
        float captureInterval = 1f / captureFPS;
        float splineLength = splinePath.CalculateLength(); // Получаем общую длину пути

        // Пока не доехали до конца сплайна
        while (isRecording && (totalFrames == 0 || frameCounter < totalFrames) && currentDistance <= splineLength)
        {
            // 1. Вычисляем нормализованное время (от 0 до 1) на сплайне
            float t = currentDistance / splineLength;

            // 2. Сдвигаем объект и поворачиваем его
            movingObject.position = splinePath.EvaluatePosition(t);

            if (rotateWithSpline)
            {
                // Находим касательную (направление вперед) и вектор "вверх" 
                float3 tangent = splinePath.EvaluateTangent(t);
                float3 upVector = splinePath.EvaluateUpVector(t);

                if (math.lengthsq(tangent) > 0.001f)
                {
                    movingObject.rotation = Quaternion.LookRotation(tangent, upVector);
                }
            }

            captureCamera.Render();
            // 3. Ждём конца кадра (чтобы рендер завершился)
            yield return new WaitForEndOfFrame();

            // 4. Сохраняем кадр
            SaveFrame(frameCounter);
            frameCounter++;

            // Увеличиваем дистанцию для следующего шага
            currentDistance += moveSpeed * captureInterval;

            // 5. Ждём интервал до следующего кадра
            yield return new WaitForSeconds(captureInterval);
        }

        Debug.Log($" Запись завершена. Всего кадров: {frameCounter}");
        isRecording = false;
    }



    void SaveFrame(int index)
    {
        // Читаем пиксели из RenderTexture 
        RenderTexture.active = renderTexture;
        Texture2D tex = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        tex.Apply();
        RenderTexture.active = null;

        // Конвертируем в PNG 
        byte[] bytes = tex.EncodeToPNG();
        Destroy(tex);

        // Сохраняем файл с нумерованным именем 
        string fileName = $"{fileNamePrefix}_{index:D4}.png";
        string filePath = Path.Combine(savePath, fileName);
        File.WriteAllBytes(filePath, bytes);

        // Каждые 30 кадров выводим в консоль
        if (index % 30 == 0) Debug.Log($"Сохранён кадр {index}: {filePath}");
    }



    [ContextMenu("Сделать тестовый кадр")]
    public void TestCapture()
    {
        if (renderTexture != null) SaveFrame(-1);
    }
}
